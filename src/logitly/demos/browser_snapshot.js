(() => {
  if (!document.body) return null;

  const cache = window.__logitlyBrowser ||= {
    ids: new WeakMap(),
    nodes: new Map(),
    next: 1,
  };

  const identity = (element) => {
    if (!cache.ids.has(element)) cache.ids.set(element, cache.next++);
    const id = cache.ids.get(element);
    cache.nodes.set(id, element);
    return id;
  };

  for (const [id, element] of cache.nodes) {
    if (!element.isConnected) cache.nodes.delete(id);
  }

  const visible = (element) => {
    if (!element || element.closest('[aria-hidden="true"],[inert]')) return false;
    if (!element.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    const rect = element.getBoundingClientRect();
    const x = rect.x + rect.width / 2;
    const y = rect.y + rect.height / 2;
    return rect.width > 0 && rect.height > 0 && x >= 0 && y >= 0 &&
      x < innerWidth && y < innerHeight;
  };

  const safe = (element) => !['password', 'file', 'hidden'].includes(element.type);

  const accessibleName = (element, seen = new Set()) => {
    if (!element || seen.has(element)) return '';
    seen.add(element);
    const labelled = (element.getAttribute('aria-labelledby') || '')
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => accessibleName(document.getElementById(id), seen))
      .filter(Boolean)
      .join(' ');
    return labelled || element.getAttribute('aria-label') ||
      [...(element.labels || [])].map((label) => accessibleName(label, seen)).filter(Boolean).join(' ') ||
      (['button', 'submit', 'reset'].includes(element.type) ? element.value : '') ||
      element.getAttribute('alt') ||
      (element.tagName === 'INPUT' ? '' : [...element.childNodes].map((node) => {
        if (node.nodeType === Node.TEXT_NODE) return node.textContent;
        if (node.nodeType === Node.ELEMENT_NODE && node.getAttribute('aria-hidden') !== 'true') {
          return accessibleName(node, seen);
        }
        return '';
      }).join(' ').trim()) ||
      element.getAttribute('title') || element.getAttribute('placeholder') || '';
  };

  const supportedRoles = [
    'button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem',
    'menuitemradio', 'option', 'gridcell', 'combobox', 'textbox',
    'searchbox', 'spinbutton',
  ];
  const selector = 'a[href],button,input,textarea,select,summary,[contenteditable="true"],' +
    supportedRoles.map((role) => `[role="${role}"]`).join(',');

  const role = (element) => {
    const explicit = element.getAttribute('role');
    if (supportedRoles.includes(explicit)) return explicit;
    if (element.tagName === 'BUTTON' || element.tagName === 'SUMMARY') return 'button';
    if (element.tagName === 'A') return 'link';
    if (element.tagName === 'SELECT') return 'combobox';
    if (element.tagName === 'TEXTAREA' || element.isContentEditable) return 'textbox';
    if (element.tagName === 'INPUT') {
      if (['checkbox', 'radio'].includes(element.type)) return element.type;
      if (['button', 'submit', 'reset', 'image'].includes(element.type)) return 'button';
      if (element.type === 'search') return 'searchbox';
      if (element.type === 'number') return 'spinbutton';
      if (['text', 'email', 'url', 'tel'].includes(element.type)) return 'textbox';
    }
    return null;
  };

  const actions = [];
  for (const element of document.querySelectorAll(selector)) {
    if (!safe(element) || !visible(element) || element.matches(':disabled') ||
        element.closest('[aria-disabled="true"]')) continue;
    const elementRole = role(element);
    if (!elementRole) continue;
    const rect = element.getBoundingClientRect();
    const base = {
      node: identity(element),
      role: elementRole,
      label: accessibleName(element) || elementRole,
      rect: {x: rect.x, y: rect.y, w: rect.width, h: rect.height},
    };
    for (const key of ['checked', 'selected', 'expanded']) {
      const value = element.getAttribute(`aria-${key}`);
      if (value !== null) base[key] = value;
    }
    if (['checkbox', 'radio'].includes(element.type)) base.checked = String(element.checked);

    if (element.tagName === 'SELECT') {
      for (const option of element.options) {
        if (option.selected || option.disabled || option.closest('optgroup[disabled]')) continue;
        actions.push({
          ...base,
          kind: 'select',
          value: option.value,
          current_value: [...element.selectedOptions].map((item) => item.label).join(', '),
          label: `${base.label} → ${option.label}`,
        });
      }
    } else {
      const editable = !element.readOnly && element.getAttribute('aria-readonly') !== 'true' &&
        (['textbox', 'searchbox', 'spinbutton'].includes(elementRole) ||
          (elementRole === 'combobox' && ['INPUT', 'TEXTAREA'].includes(element.tagName)));
      const value = 'value' in element ? String(element.value) :
        element.isContentEditable || elementRole === 'combobox' ? element.innerText.trim() : '';
      actions.push({...base, kind: editable ? 'fill' : 'click', value});
      if (editable) actions.push({...base, kind: 'click', value, label: `Open ${base.label}`});
    }
  }

  const words = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  let node;
  let length = 0;
  while ((node = walker.nextNode()) && length < 4000) {
    const value = node.textContent.trim();
    const parent = node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
    range.selectNodeContents(node);
    const rect = range.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < innerHeight &&
        rect.right > 0 && rect.left < innerWidth) {
      words.push(value);
      length += value.length;
    }
  }

  const text = words.join('\n').slice(0, 4000);
  const height = document.documentElement.scrollHeight;
  const marker = JSON.stringify([
    performance.timeOrigin, location.href, scrollX, scrollY, innerWidth, innerHeight,
    document.title, text,
    actions.map(({rect, ...action}) => action),
    [...document.querySelectorAll('input,textarea,select')].filter(safe)
      .map((element) => [identity(element), element.value, element.checked, element.selectedIndex]),
  ]);
  cache.marker = () => marker;

  const omitted_actions = Math.max(0, actions.length - 60);
  actions.splice(60);
  actions.forEach((action, index) => { action.id = `e${index + 1}`; });
  if (scrollY + innerHeight < height - 2) {
    actions.push({id: 'scroll_down', kind: 'scroll', label: 'Scroll down', delta: 560});
  }
  if (scrollY > 0) {
    actions.push({id: 'scroll_up', kind: 'scroll', label: 'Scroll up', delta: -560});
  }
  actions.push({id: 'wait', kind: 'wait', label: 'Wait for the page to update'});

  return {
    url: location.href,
    title: document.title,
    text,
    actions,
    marker,
    scroll: {y: scrollY, height},
    omitted_actions,
  };
})()
