"use strict";

(() => {
  const guideHash = "#technique-guide";
  const familyPrefix = guideHash + "/";
  const families = new Set(["total", "saxs", "waxs", "tomo", "optics", "attenuation"]);

  function parse(hash) {
    if (hash === guideHash) return {family: ""};
    if (typeof hash !== "string" || !hash.startsWith(familyPrefix)) return null;
    const family = hash.slice(familyPrefix.length);
    return {family: families.has(family) ? family : ""};
  }

  function hashFor(family) {
    return families.has(family) ? familyPrefix + family : guideHash;
  }

  globalThis.WorkbenchGuideFamilyLinks = {parse, hashFor};
})();
