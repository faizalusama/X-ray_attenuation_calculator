"use strict";
(() => {
  const scenes = {
    total: {
      title: "Fourier transform interval", options: ["Lower Qmax", "Higher Qmax"],
      notes: ["The highlighted samples enter the transform. A lower upper cutoff discards more high-Q information.",
        "More of this same measured interval enters the transform. Finite-range effects still depend on the window and data."],
      draw(second) {
        const end = second ? 251 : 159;
        const points = Array.from({length: 11}, (_, i) => {
          const x = 44 + 23 * i;
          return '<circle cx="' + x + '" cy="65" r="4" class="' + (x <= end ? "gd-spot" : "gd-unused") + '"/>';
        }).join("");
        return '<rect x="44" y="41" width="' + (end - 44) + '" height="47" class="gd-fill"/>' +
          path("M22 65H295") + points + path("M" + end + " 35v58", "gd-accent") +
          label(44, 111, "Qmin") + label(end, 24, "Qmax") + label(280, 111, "Q");
      }
    },
    saxs: {
      title: "Detector distance and coverage", options: ["Near detector", "Far detector"],
      notes: ["The outer detector radius subtends a larger scattering angle at the shorter distance.",
        "Moving the same detector farther away reduces the angle and Q reached at its edge for a fixed wavelength."],
      draw(second) {
        const edge = second ? 262 : 165;
        return path("M18 78H285", "gd-axis") + path("M20 78H75") +
          path("M75 78L" + edge + " 38V78Z", "gd-fill") + path("M75 78L" + edge + " 38", "gd-accent") +
          path("M" + edge + " 37v78") + spot(75,78) + label(75,104,"Sample") + label(edge,24,"Detector");
      }
    },
    waxs: {
      title: "Scattering angle and spacing", options: ["Lower angle", "Higher angle"],
      notes: ["The outgoing ray is closer to the direct beam. At fixed wavelength this angle corresponds to lower Q and larger d.",
        "The higher scattering angle corresponds to higher Q and smaller d at the same wavelength."],
      draw(second) {
        const y = second ? 23 : 67;
        return path("M18 94H279","gd-axis") + path("M20 94H96") + path("M250 17v91") +
          path("M96 94L250 " + y,"gd-accent") + spot(96,94) + spot(250,y) +
          label(92,116,"Sample") + label(270,116,"Detector");
      }
    },
    tomo: {
      title: "Angular sampling", options: ["Fewer views", "More views"],
      notes: ["Each line represents one parallel-beam viewing direction through the same support. Large angular gaps separate neighboring views.",
        "More evenly spaced views reduce angular gaps over the same half-turn. The detector pixel pitch still limits spatial sampling."],
      draw(second) {
        const n = second ? 12 : 4;
        const rays = Array.from({length:n}, (_, i) =>
          '<g transform="rotate(' + (180*i/n) + ' 153 61)">' + path("M79 61h148", i ? "gd-secondary" : "gd-accent") + '</g>'
        ).join("");
        return '<circle cx="153" cy="61" r="36" class="gd-fill"/>' + rays +
          '<circle cx="153" cy="61" r="36" class="gd-outline"/>' + spot(153,61);
      }
    },
    optics: {
      title: "Specular reflection geometry", options: ["Shallower angle", "Steeper angle"],
      notes: ["Incident and reflected rays meet the surface at equal grazing angles in the specular direction.",
        "Both angles increase together. Reflectivity and penetration depth also depend on the material and photon energy."],
      draw(second) {
        const y = second ? 20 : 63;
        return '<rect x="20" y="84" width="272" height="25" class="gd-fill"/>' + path("M20 84H292") +
          path("M43 " + y + "L152 84L261 " + y,"gd-accent") + spot(152,84) +
          label(48,y-9,"Incident") + label(248,y-9,"Reflected") + label(150,104,"Surface");
      }
    },
    attenuation: {
      title: "Beam path through a slab", options: ["Normal incidence", "Oblique incidence"],
      notes: ["At normal incidence the path through the slab equals its normal thickness.",
        "At oblique incidence the beam travels farther through the same slab. For positive attenuation coefficient this reduces narrow-beam transmission."],
      draw(second) {
        const y1 = second ? 88 : 68, y2 = second ? 48 : 68;
        return '<rect x="114" y="17" width="82" height="91" class="gd-fill"/>' + path("M114 17v91M196 17v91") +
          path("M114 68H196","gd-axis") + path(second ? "M53 118L114 88M196 48L263 15" : "M32 68H114M196 68H275") +
          path("M114 " + y1 + "L196 " + y2,"gd-accent") + spot(114,y1) + spot(196,y2) + label(155,33,"Slab");
      }
    }
  };
  // Geometry is illustrative. The technique calculators own all quantitative results.
  function path(d, cls = "gd-outline") { return '<path class="' + cls + '" d="' + d + '"/>'; }
  function spot(x,y) { return '<circle class="gd-spot" cx="' + x + '" cy="' + y + '" r="4"/>'; }
  function label(x,y,text) { return '<text x="' + x + '" y="' + y + '" text-anchor="middle">' + text + '</text>'; }

  globalThis.WorkbenchGuideDiagram = function (id) {
    const scene = scenes[id];
    if (!scene) return null;
    const panel = document.createElement("details");
    panel.className = "guide-diagram";
    const heading = document.createElement("summary");
    heading.textContent = scene.title;
    const figure = document.createElement("figure");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 -16 310 157");
    svg.setAttribute("role", "img");
    const caption = document.createElement("figcaption");
    caption.setAttribute("aria-live", "polite");
    const choices = document.createElement("div");
    choices.className = "gd-choices";
    choices.setAttribute("role", "group");
    choices.setAttribute("aria-label", scene.title);
    const buttons = scene.options.map((option,index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = option;
      button.addEventListener("click", () => select(index));
      choices.append(button);
      return button;
    });
    function select(index) {
      buttons.forEach((button,i) => button.setAttribute("aria-pressed", String(index === i)));
      svg.innerHTML = scene.draw(index === 1);
      svg.setAttribute("aria-label", scene.title + ". " + scene.options[index] + ". " + scene.notes[index]);
      caption.textContent = scene.notes[index];
    }
    select(0);
    figure.append(choices,svg,caption);
    panel.append(heading,figure);
    return panel;
  };
})();

