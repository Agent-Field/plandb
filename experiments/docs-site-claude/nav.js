/* PlanDB Docs — Shared Navigation */
(function () {
  'use strict';

  var pages = [
    { href: 'getting-started.html', label: 'Getting Started' },
    { href: 'cli-reference.html', label: 'CLI Reference' },
    { href: 'architecture.html', label: 'Architecture' },
    { href: 'playground.html', label: 'Playground' }
  ];

  var currentPath = window.location.pathname.split('/').pop() || 'index.html';

  /* Build nav DOM */
  var nav = document.createElement('nav');
  nav.className = 'nav';
  nav.setAttribute('role', 'navigation');
  nav.setAttribute('aria-label', 'Main navigation');

  var inner = document.createElement('div');
  inner.className = 'nav-inner';

  /* Brand */
  var brand = document.createElement('a');
  brand.href = 'index.html';
  brand.className = 'nav-brand';
  brand.textContent = 'PlanDB';
  inner.appendChild(brand);

  /* Links */
  var ul = document.createElement('ul');
  ul.className = 'nav-links';
  ul.id = 'nav-links';

  pages.forEach(function (page) {
    var li = document.createElement('li');
    li.style.listStyle = 'none';
    var a = document.createElement('a');
    a.href = page.href;
    a.textContent = page.label;
    if (currentPath === page.href) {
      a.classList.add('active');
    }
    li.appendChild(a);
    ul.appendChild(li);
  });

  inner.appendChild(ul);

  /* Mobile toggle */
  var toggle = document.createElement('button');
  toggle.className = 'nav-toggle';
  toggle.setAttribute('aria-label', 'Toggle navigation');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.innerHTML = '&#9776;';

  toggle.addEventListener('click', function () {
    var isOpen = ul.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(isOpen));
    toggle.innerHTML = isOpen ? '&#10005;' : '&#9776;';
  });

  inner.appendChild(toggle);
  nav.appendChild(inner);

  /* Insert */
  document.body.insertBefore(nav, document.body.firstChild);

  /* Close mobile menu on link click */
  ul.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') {
      ul.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.innerHTML = '&#9776;';
    }
  });

  /* Close mobile menu on outside click */
  document.addEventListener('click', function (e) {
    if (!nav.contains(e.target) && ul.classList.contains('open')) {
      ul.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.innerHTML = '&#9776;';
    }
  });
})();
