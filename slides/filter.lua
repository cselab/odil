colors = {
  red = 'color0', --red
  MyDarkGreen = 'color1', --green
  blue = 'color2', --blue
  color0 = 'color0', --red
  color1 = 'color1', --green
  color2 = 'color2', --blue
  color3 = 'color3',
  color4 = 'color4',
  color5 = 'color5', --orange
}

function TableToString(content)
  res = ""
  for k,v in pairs(content) do
    res = res .. tostring(v) .. '\n' 
  end
  return res
end

function Span(el)
  for tex, class in pairs(colors) do
    if el.content[1].text == 'COLOR[' .. tex .. ']' then
      el.classes[1] = class
      el.content[1] = ""
      assert(el.content[2] == pandoc.Space()
             or el.content[2] == pandoc.SoftBreak()
             , TableToString(el.content))
      el.content[2] = ""
    end
  end

  return el
end

if FORMAT:match 'markdown' then
  function Image(el)
    el.caption = ''
    return el
  end
end

if FORMAT:match 'revealjs' then
  function Image(el)
    el.attributes.style = 'margin:auto; display: block;'
    if not el.attributes.height and not el.attributes.height then
      el.attributes.height = '250'
    end
    return el
  end

  function BulletList(el)
    for i, item in ipairs(el.content) do
      local first = item[1]
      if first and first.t == 'Para' then
        el.content[i][1] = first.content
      end
    end
    return el
  end

  function Math(el)
    for tex,class in pairs(colors) do
      el.text = el.text:gsub('{COLOR%[' .. tex .. '%]', '\\htmlClass{'..class..'}{')
      el.text = el.text:gsub('COLOR%[' .. tex .. '%]', '\\htmlClass{'..class..'}')
    end
    if el.mathtype == 'InlineMath' then
      return pandoc.Str('$'.. el.text ..'$')
    else
      return pandoc.Str('\n\\[\n'.. el.text ..'\n\\]\n')
    end
  end
end
