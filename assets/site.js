(() => {
  'use strict';

  const navigation = document.querySelector('details.section-navigation');
  if (navigation) navigation.open = window.matchMedia('(min-width: 1501px)').matches;
  const links = Array.from(navigation?.querySelectorAll('a[href^="#"]') || []);
  const sections = Array.from(document.querySelectorAll('[data-section][id]'));

  const setActiveSection = (id) => {
    links.forEach((link) => {
      if (link.hash === `#${id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };

  links.forEach((link) => {
    link.addEventListener('click', () => {
      setActiveSection(link.hash.slice(1));
      if (window.matchMedia('(max-width: 1500px)').matches) {
        navigation.open = false;
        const target = document.getElementById(link.hash.slice(1));
        if (target) {
          const previousTabIndex = target.getAttribute('tabindex');
          target.setAttribute('tabindex', '-1');
          target.focus({ preventScroll: true });
          target.addEventListener('blur', () => {
            if (previousTabIndex === null) target.removeAttribute('tabindex');
            else target.setAttribute('tabindex', previousTabIndex);
          }, { once: true });
        }
      }
    });
  });

  if (sections.length) {
    let framePending = false;
    const updateActiveSection = () => {
      framePending = false;
      const marker = Math.max(64, window.innerHeight * 0.25);
      let active = sections[0];
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= marker) active = section;
      }
      // A short final section cannot always reach the marker: scrolling is
      // clamped at the end of the document. Still highlight Citation there.
      if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
        active = sections[sections.length - 1];
      }
      setActiveSection(active.id);
    };
    const scheduleUpdate = () => {
      if (framePending) return;
      framePending = true;
      window.requestAnimationFrame(updateActiveSection);
    };
    window.addEventListener('scroll', scheduleUpdate, { passive: true });
    window.addEventListener('resize', scheduleUpdate);
    window.addEventListener('hashchange', scheduleUpdate);
    window.addEventListener('load', scheduleUpdate);
    updateActiveSection();
  }

  const copyStatus = document.querySelector('.copy-status');
  if (copyStatus) {
    copyStatus.setAttribute('role', 'status');
    copyStatus.setAttribute('aria-live', 'polite');
  }
  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    const originalLabel = button.textContent;
    let resetLabel;
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget.replace(/^#/, ''));
      if (!target) return;
      if (copyStatus) copyStatus.textContent = '';
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(target.textContent.trim());
        button.textContent = 'Copied!';
        if (copyStatus) copyStatus.textContent = 'Copied to clipboard.';
      } catch {
        const range = document.createRange();
        range.selectNodeContents(target);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
        button.textContent = 'Text selected';
        if (copyStatus) copyStatus.textContent = 'Text selected. Use your keyboard or context menu to copy.';
      }
      clearTimeout(resetLabel);
      resetLabel = setTimeout(() => { button.textContent = originalLabel; }, 2500);
    });
  });

  const dialog = document.getElementById('figure-dialog');
  const dialogImage = document.getElementById('figure-dialog-image');
  const dialogCaption = document.getElementById('figure-dialog-caption');
  if (!dialog || !dialogImage || typeof dialog.showModal !== 'function') return;

  document.querySelectorAll('a[data-lightbox]').forEach((link) => {
    link.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const sourceImage = link.querySelector('img');
      dialogImage.src = link.href;
      dialogImage.alt = sourceImage?.alt || '';
      if (dialogCaption) {
        dialogCaption.textContent = link.dataset.caption || link.closest('figure')?.querySelector('figcaption')?.textContent || sourceImage?.alt || '';
      }
      dialog.showModal();
      event.preventDefault();
    });
  });
  dialog.querySelectorAll('[data-close-dialog]').forEach((button) => {
    button.addEventListener('click', () => dialog.close());
  });
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
})();
