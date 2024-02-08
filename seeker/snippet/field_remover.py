#date: 2024-02-08T17:04:00Z
#url: https://api.github.com/gists/dc7ac6656770fc593c0d9a4164b148f3
#owner: https://api.github.com/users/petrov-aleksandr

from pprint import pprint


class KeyDoesNotExist:
    pass


def remove_fields(selector: str, dashboard: dict):
    def traverse(selectors: list[str], obj: dict):
        selector = selectors[0]
        if len(selectors) == 1:
            if selector in obj:
                print(f"removing {selector}: {obj[selector]}")
                del obj[selector]
            return
        item = obj.get(selector, KeyDoesNotExist)
        if item == KeyDoesNotExist:
            return
        if isinstance(item, dict):
            traverse(selectors[1:], item)
        elif isinstance(item, list):
            for i in item:
                traverse(selectors[1:], i)
        else:
            raise ValueError(f"value {item} is not traversable. Selectors: {selectors}")

    selectors = selector.split(".")
    traverse(selectors, dashboard)
    return dashboard


if __name__ == '__main__':
    from yaml import load, dump, Loader, Dumper

    with open('in_copy.yaml') as r:
        d = load(r, Loader)

    fields_to_remove = [
        "dashboard.annotations",
        "dashboard.description",
        "dashboard.fiscalYearStartMonth",
        "dashboard.graphTooltip",
        "dashboard.liveNow",
        "dashboard.refresh",
        "dashboard.schemaVersion",
        "dashboard.style",
        "dashboard.timepicker",
        "dashboard.id",
        "dashboard.uid",
        "dashboard.version",
        "dashboard.weekStart",
        "dashboard.panels.gridPos",
        "dashboard.panels.id",
        "dashboard.panels.datasource",
        "dashboard.panels.targets",
        "dashboard.panels.panels.fieldConfig.defaults.custom.axisColorMode",
        "dashboard.panels.panels.fieldConfig.defaults.custom.axisCenteredZero",
        "dashboard.panels.panels.options.timezone",
        "dashboard.panels.panels.options.alignValue",
        "dashboard.panels.panels.options.mergeValues",
        "dashboard.panels.panels.options.rowHeight",
        "dashboard.templating.allFormat",
        "dashboard.templating.datasource",
        "dashboard.templating.current.selected",
        "dashboard.templating.skipUrlSync",
        "dashboard.templating.queryValue",

    ]

    d = {"dashboard": d}

    if "list" in d["dashboard"]["templating"]:
        d["dashboard"]["templating"] = d["dashboard"]["templating"]["list"]

    if "list" in d["dashboard"]["annotations"]:
        d["dashboard"]["annotations"] = d["dashboard"]["annotations"]["list"]
    #

    for row in d["dashboard"]["panels"]:
        for panel in row["panels"]:
            if panel.get("fieldConfig", {}).get("defaults", {}).get("thresholds", {}).get("steps", {}):
                for step in panel.get("fieldConfig", {}).get("defaults", {}).get("thresholds", {}).get("steps", {}):
                    if 'value' not in step:
                        step["value"] = 80

    for template in d["dashboard"]["templating"]:
        if template["type"] in ("query", "custom"):
            template["current"]["value"] = [template["current"]["value"]]
        if template["type"] != "custom":
            del template["options"]
        if template["type"] == "textbox" and "current" in template:
            del template["current"]

    for f in fields_to_remove:
        remove_fields(f, d)

    print("panel types:")
    for rows in d["dashboard"]["panels"]:
        for p in rows["panels"]:
            print(p["type"])





    with open('out.yaml', 'w') as w:
        dump(d, w, Dumper=Dumper)
