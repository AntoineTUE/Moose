window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { '[+]': ['mhchem'] }
  },
  loader: { load: ['[tex]/mhchem'] },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(async () => {
  if (MathJax.startup?.promise) {
    await MathJax.startup.promise;
  }

  if (typeof MathJax.typesetPromise === "function") {
    await MathJax.typesetPromise();
  }
});
