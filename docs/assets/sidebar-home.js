/* Jupyter Book 1.x drops the root document from the sidebar, while the reference site
   lists it as "Welcome! 👋" above the first group. The theme's sidebar template cannot be
   overridden from this book, so the entry is prepended here. */
document.addEventListener("DOMContentLoaded", function () {
  var nav = document.querySelector(".bd-toc-item.navbar-nav");
  if (!nav || nav.querySelector(".apt-home-link")) {
    return;
  }

  var inDocs = window.location.pathname.indexOf("/docs/") !== -1;
  var onHome = !inDocs && /README\.html$|\/$/.test(window.location.pathname);
  var highlight = onHome ? ' style="color:#0a5f91;font-weight:600"' : "";
  var list = document.createElement("ul");
  list.className = "bd-navbar-elements navbar-nav";
  list.innerHTML =
    '<li class="nav-item apt-home-link">' +
    '<a class="nav-link nav-internal"' + highlight + ' href="' +
    (inDocs ? "../README.html" : "README.html") + '">Welcome! 👋</a></li>';
  nav.insertBefore(list, nav.firstChild);
});
