"use strict";
(() => {
  const stages = ["Plan", "Measure", "Interpret"];

  function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function field(id, labelText, control) {
    const label = element("label", labelText);
    control.id = id;
    control.setAttribute("aria-controls", "guide-results");
    label.htmlFor = id;
    const wrapper = element("div");
    wrapper.append(label, control);
    return wrapper;
  }

  function destination(target) {
    if (!target) return null;
    const [kind, value] = target.split(":");
    let label, activate;
    if (kind === "tool") {
      const select = document.getElementById("st-tool");
      const option = select && Array.from(select.options).find(item => item.value === value);
      if (!option) return null;
      label = option.label;
      activate = () => {
        select.value = value;
        select.dispatchEvent(new Event("change", {bubbles: true}));
        select.focus();
      };
    } else if (kind === "tab") {
      const tab = document.getElementById("tab-" + value);
      if (!tab || tab.getAttribute("role") !== "tab") return null;
      label = tab.textContent.trim();
      activate = () => { tab.click(); tab.focus(); };
    } else {
      return null;
    }
    const button = element("button", "Open " + label);
    button.type = "button";
    button.dataset.guideTarget = target;
    button.addEventListener("click", activate);
    return button;
  }

  function buildTopic(topic) {
    const details = element("details");
    details.dataset.guideStage = topic.stage;
    const summary = element("summary");
    summary.textContent = topic.title;

    const body = element("div");
    body.className = "guide-topic-body";
    const check = element("p");
    check.append(element("strong", "Check "), topic.check);
    const actions = element("div");
    actions.className = "guide-topic-actions";
    const source = element("a", topic.source.label);
    source.href = topic.source.url;
    const sourceLine = element("div");
    sourceLine.append("Source · ", source);
    actions.append(sourceLine);
    const button = destination(topic.target);
    if (button) actions.append(button);
    body.append(element("p", topic.guidance), check, actions);
    details.append(summary, body);
    return details;
  }

  function buildFamily(family) {
    const section = element("section");
    section.dataset.guideFamily = family.id;
    const header = element("header");
    const title = element("h3", family.title);
    title.id = "guide-heading-" + family.id;
    section.setAttribute("aria-labelledby", title.id);
    header.append(title, element("p", family.intro));
    const diagram = globalThis.WorkbenchGuideDiagram?.(family.id);
    if (diagram instanceof Node) header.append(diagram);
    if (family.id === "tomo") header.append(globalThis.WorkbenchGuideVolume());
    section.append(header);

    const groups = stages.map(stage => {
      const group = element("div");
      group.className = "guide-workflow-stage";
      group.append(element("h4", {Plan: "Plan the experiment", Measure: "Collect data",
        Interpret: "Interpret results"}[stage]));
      const topics = family.topics.filter(topic => topic.stage === stage).map(topic => {
        const details = buildTopic(topic);
        group.append(details);
        const text = [family.id, family.title, family.intro, topic.title, topic.stage,
          topic.guidance, topic.check, topic.source.label].join(" ").toLowerCase();
        return {details, text};
      });
      section.append(group);
      return {group, stage, topics};
    });
    return {section, groups};
  }

  function hasText(value) {
    return typeof value === "string" && value.trim().length > 0;
  }

  function validFamily(family) {
    return family && [family.id, family.title, family.intro].every(hasText) &&
      Array.isArray(family.topics) && family.topics.length > 0 &&
      family.topics.every(topic => topic && stages.includes(topic.stage) &&
        [topic.title, topic.guidance, topic.check, topic.source?.label].every(hasText) &&
        /^https?:\/\//i.test(topic.source.url) && (topic.target == null || hasText(topic.target)));
  }

  function install() {
    const root = document.querySelector(".technique-guide #technique-explorer");
    const families = globalThis.WorkbenchGuideFamilies;
    if (!root || !Array.isArray(families) || !families.length || !families.every(validFamily)) return;

    const search = element("input");
    search.type = "search";
    const familySelect = element("select");
    familySelect.add(new Option("All families", ""));
    for (const family of families) familySelect.add(new Option(family.title, family.id));
    const stageSelect = element("select");
    stageSelect.add(new Option("All stages", ""));
    for (const stage of stages) stageSelect.add(new Option(stage, stage));
    const filters = element("div");
    filters.className = "guide-filters";
    filters.setAttribute("role", "group");
    filters.setAttribute("aria-label", "Filter technique topics");
    filters.append(field("guide-search", "Search topics", search),
      field("guide-family", "Technique family", familySelect), field("guide-stage", "Stage", stageSelect));

    const count = element("p");
    count.id = "guide-count";
    count.setAttribute("role", "status");
    count.setAttribute("aria-live", "polite");
    count.setAttribute("aria-atomic", "true");
    const reset = element("button", "Reset filters");
    reset.id = "guide-reset";
    reset.type = "button";
    const status = element("div");
    status.className = "guide-status";
    status.append(count, reset);
    const results = element("div");
    results.id = "guide-results";
    const empty = element("p", "Try a different search or reset the filters to see all topics.");
    empty.id = "guide-empty";
    const additions = globalThis.WorkbenchGuideFieldChecks;
    const groups = families.map(family => {
      const combined = {...family, topics: [...family.topics, ...(additions?.[family.id] || [])]};
      return buildFamily(validFamily(combined) ? combined : family);
    });
    results.append(empty, ...groups.map(group => group.section));

    function filter() {
      const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
      let topicCount = 0, familyCount = 0;
      for (const {section, groups: workflow} of groups) {
        const matchesFamily = !familySelect.value || section.dataset.guideFamily === familySelect.value;
        let visible = 0;
        for (const {group, stage, topics} of workflow) {
          const matchesStage = matchesFamily && (!stageSelect.value || stage === stageSelect.value);
          let stageCount = 0;
          for (const {details, text} of topics) {
            const matches = matchesStage && terms.every(term => text.includes(term));
            details.hidden = !matches;
            if (matches) stageCount++;
          }
          group.hidden = stageCount === 0;
          visible += stageCount;
        }
        section.hidden = visible === 0;
        if (visible) familyCount++;
        topicCount += visible;
      }
      empty.hidden = topicCount !== 0;
      count.textContent = topicCount + (topicCount === 1 ? " topic" : " topics") +
        " in " + familyCount + (familyCount === 1 ? " family" : " families");
    }
    search.addEventListener("input", filter);
    familySelect.addEventListener("change", filter);
    stageSelect.addEventListener("change", filter);
    reset.addEventListener("click", () => {
      search.value = familySelect.value = stageSelect.value = "";
      filter();
      search.focus();
    });
    filter();
    root.replaceChildren(filters, status, results);
    root.hidden = false;
    const fallback = root.closest(".technique-guide").querySelector("#technique-guide-fallback");
    if (fallback) fallback.hidden = true;
  }

  document.addEventListener("DOMContentLoaded", install, {once: true});
})();
