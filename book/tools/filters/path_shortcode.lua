-- Quarto filter to handle {{< path >}} shortcodes
-- Converts {{< path images/logo.png >}} to /images/logo.png

function Str(elem)
  -- Handle path shortcodes in text strings
  local text = elem.text
  if string.find(text, "{{<") then
    -- Match and replace path shortcodes
    local new_text = string.gsub(text, "{{<%s*path%s+([^>]+)%s*>}}", function(path)
      -- Clean up the path (remove quotes)
      local clean_path = string.gsub(path, "^['\"]", ""):gsub("['\"]$", "")
      clean_path = string.gsub(clean_path, "^%s*", ""):gsub("%s*$", "")  -- trim whitespace
      -- Return absolute path
      return "/" .. clean_path
    end)
    
    if new_text ~= text then
      return pandoc.Str(new_text)
    end
  end
  return elem
end

function RawInline(elem)
  -- Handle path shortcodes in raw inline elements
  if elem.format == "html" then
    local text = elem.text
    if string.find(text, "{{<") then
      local new_text = string.gsub(text, "{{<%s*path%s+([^>]+)%s*>}}", function(path)
        local clean_path = string.gsub(path, "^['\"]", ""):gsub("['\"]$", "")
        clean_path = string.gsub(clean_path, "^%s*", ""):gsub("%s*$", "")
        return "/" .. clean_path
      end)
      
      if new_text ~= text then
        return pandoc.RawInline(elem.format, new_text)
      end
    end
  end
  return elem
end

function RawBlock(elem)
  -- Handle path shortcodes in raw block elements  
  if elem.format == "html" then
    local text = elem.text
    if string.find(text, "{{<") then
      local new_text = string.gsub(text, "{{<%s*path%s+([^>]+)%s*>}}", function(path)
        local clean_path = string.gsub(path, "^['\"]", ""):gsub("['\"]$", "")
        clean_path = string.gsub(clean_path, "^%s*", ""):gsub("%s*$", "")
        return "/" .. clean_path
      end)
      
      if new_text ~= text then
        return pandoc.RawBlock(elem.format, new_text)
      end
    end
  end
  return elem
end