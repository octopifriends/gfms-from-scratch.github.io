-- Quarto Lua filter: tangle code blocks to files
-- Usage in qmd code block options:
--   ```{python}
--   #| tangle: ../geogfm/module/file.py
--   #| tangle-append: true   (optional; default false)
--   #| tangle-marker: "# --- from: path.qmd ---" (optional header)
--   code...
--   ```

local quarto = quarto
local pandoc = pandoc

-- Utility: ensure directory exists
local lfs = nil
pcall(function() lfs = require('lfs') end)

local function dirname(path)
  return path:match("^(.*)/[^/]-$") or "."
end

local function makedirs(path)
  if not lfs then return end
  local parts = {}
  for part in path:gmatch("[^/]+") do table.insert(parts, part) end
  local acc = ""
  for i, p in ipairs(parts) do
    acc = (i == 1) and p or (acc .. "/" .. p)
    if lfs.attributes(acc, 'mode') == nil then
      lfs.mkdir(acc)
    end
  end
end

-- Write file with optional append
local function write_file(path, content, append)
  local mode = append and 'a' or 'w'
  local f, err = io.open(path, mode)
  if not f then
    -- try to create parent dirs
    makedirs(dirname(path))
    f, err = io.open(path, mode)
  end
  assert(f, err or ("cannot open " .. path))
  f:write(content)
  f:write("\n")
  f:close()
end

-- Resolve path relative to project root (book directory)
-- Config captured from document metadata
local config = {
  root_mode = 'auto',  -- 'auto'|'repo'|'path'
  root_path = nil
}

-- Find repo root (looks for .git directory)
local function find_repo_root(start_dir)
  if not lfs then return nil end
  local dir = start_dir
  while dir and dir ~= '' do
    local gitdir = dir .. '/.git'
    if lfs.attributes(gitdir, 'mode') == 'directory' then
      return dir
    end
    local parent = dir:match("^(.*)/[^/]-$")
    if not parent or parent == dir then break end
    dir = parent
  end
  return nil
end

local function resolve_path(rel)
  -- Absolute path as-is
  if rel:match('^/') then return rel end

  local input_file = PANDOC_STATE.input_files[1] or ''
  local doc_dir = pandoc.path and pandoc.path.directory(input_file) or '.'
  local project_base = doc_dir

  -- Determine base per config
  if config.root_mode == 'repo' then
    local repo = find_repo_root(doc_dir)
    project_base = repo or doc_dir
  elseif config.root_mode == 'path' and config.root_path then
    if not config.root_path:match('^/') then
      -- relative to doc_dir
      project_base = pandoc.path.normalize(pandoc.path.join({doc_dir, config.root_path}))
    else
      project_base = config.root_path
    end
  else
    -- auto: if rel starts with ./ or ../ use doc_dir, otherwise prefer repo root
    if rel:match('^%./') or rel:match('^%../') then
      project_base = doc_dir
    else
      local repo = find_repo_root(doc_dir)
      project_base = repo or doc_dir
    end
  end

  local full = pandoc.path and pandoc.path.normalize(pandoc.path.join({project_base, rel})) or rel
  return full
end

-- Extract custom options from fenced code block attributes/comments
local function get_option_from_attr(attr, name)
  if attr and attr[name] then return attr[name] end
  return nil
end

-- Also allow knitr-style hash pipe directives inside code as fallback
local function get_option_from_code(code, name)
  local pattern = "#%|%s*" .. name .. ":%s*(.-)%s*$"
  for line in code:gmatch("[^\n]*\n?") do
    local val = line:match(pattern)
    if val then
      -- strip surrounding quotes if present
      val = val:gsub('^%s*"', ''):gsub('"%s*$', '')
      val = val:gsub("^%s*'", ''):gsub("'%s*$", '')
      return val
    end
  end
  return nil
end

return {
  -- Capture document metadata once
  {
    Pandoc = function(doc)
      local meta = doc.meta or {}
      local tr = meta["tangle-root"]
      if tr then
        if type(tr) == 'table' and tr.t then
          tr = pandoc.utils.stringify(tr)
        end
        if tr == 'repo' then
          config.root_mode = 'repo'
        elseif tr and tr ~= '' then
          config.root_mode = 'path'
          config.root_path = tr
        end
      end
      return nil
    end,

    CodeBlock = function(cb)
      -- Only tangle executable language blocks (python, r, bash, etc.)
      if not cb.classes or #cb.classes == 0 then return nil end

      local code = cb.text or ''
      local attr = cb.attributes or {}

      -- Prefer attribute-based option: tangle="path"
      local tangle_path = get_option_from_attr(attr, 'tangle') or get_option_from_code(code, 'tangle')
      if not tangle_path or tangle_path == '' then return nil end

      local append_opt = get_option_from_attr(attr, 'tangle-append') or get_option_from_code(code, 'tangle-append')
      local append = (append_opt == 'true' or append_opt == 'True' or append_opt == '1')

      local header = get_option_from_attr(attr, 'tangle-marker') or get_option_from_code(code, 'tangle-marker')
      local resolved = resolve_path(tangle_path)

      -- Compose content with optional header marker including source doc
      local src = PANDOC_STATE.input_files[1] or 'unknown'
      local id = cb.identifier or ''
      local ts = os.date('!%Y-%m-%dT%H:%M:%SZ')
      local default_marker = string.format("# --- tangle from: %s%s%s | time: %s ---", src, (id ~= '' and ' | id: ' or ''), (id ~= '' and id or ''), ts)
      local marker = header or default_marker
      local content = marker .. "\n" .. code .. "\n"

      -- Write file
      pcall(function()
        write_file(resolved, content, append)
        quarto.log.output(string.format("[tangle] wrote %s (%s)", resolved, append and 'append' or 'write'))
      end)

      return nil
    end
  }
}


