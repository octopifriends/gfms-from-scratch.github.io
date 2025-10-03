-- Quarto shortcode extension for path resolution
-- Registers a 'path' shortcode that converts relative paths to absolute paths

return {
  ['path'] = function(args, kwargs, meta)
    -- Get the path from the first argument
    if args and #args > 0 then
      local path = pandoc.utils.stringify(args[1])
      -- Clean up quotes if present
      path = path:gsub('^["\']', ''):gsub('["\']$', '')
      path = path:gsub('^%s*', ''):gsub('%s*$', '')  -- trim whitespace
      -- Return absolute path for consistent cross-directory access
      -- Use RawInline to preserve in HTML context
      return pandoc.RawInline('html', '/' .. path)
    end
    
    -- Fallback if no path provided
    return pandoc.Str('')
  end
}