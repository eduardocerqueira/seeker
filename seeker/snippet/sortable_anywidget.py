#date: 2025-05-26T16:51:26Z
#url: https://api.github.com/gists/1ca8f80088f83ea6f14bdb83ca16acb4
#owner: https://api.github.com/users/manzt

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget==0.9.18",
#     "marimo",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(genres):
    genres.value
    return


@app.cell
def _(Sortable):
    import marimo as mo

    genres = mo.ui.anywidget(
        Sortable(
            [
                "Action",
                "Comedy",
                "Drama",
                "Thriller",
                "Sci-Fi",
                "Animation",
                "Documentary",
            ]
        )
    )
    genres
    return (genres,)


@app.cell
def _():
    import typing

    import anywidget
    import traitlets

    class Sortable(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          el.classList.add("draggable-list-widget");

          let draggedItem = null;
          let draggedIndex = null;
          let dropTarget = null;
          let dropPosition = null;

          function renderList() {
            el.replaceChildren();

            let container = document.createElement("div");
            container.className = "list-container";

            model.get("value").forEach((item, index) => {
              let listItem = document.createElement("div");
              listItem.className = "list-item";
              listItem.draggable = true;
              listItem.dataset.index = index;

              let dragHandle = document.createElement("button");
              dragHandle.className = "drag-handle";
              dragHandle.innerHTML = `
                <svg width="10" height="10" viewBox="0 0 16 16">
                  <circle cx="4" cy="4" r="1"/>
                  <circle cx="12" cy="4" r="1"/>
                  <circle cx="4" cy="8" r="1"/>
                  <circle cx="12" cy="8" r="1"/>
                  <circle cx="4" cy="12" r="1"/>
                  <circle cx="12" cy="12" r="1"/>
                </svg>
              `;
              dragHandle.setAttribute("aria-label", `Reorder ${item}`);

              let label = document.createElement("span");
              label.className = "item-label";
              label.textContent = item;

              let removeButton = document.createElement("button");
              removeButton.className = "remove-button";
              removeButton.innerHTML = `
                <svg width="10" height="10" viewBox="0 0 14 14" fill="none">
                  <path d="M4 4l6 6m0-6l-6 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
              `;
              removeButton.setAttribute("aria-label", `Remove ${item}`);
              removeButton.onclick = (e) => {
                e.stopPropagation();
                removeItem(index);
              };

              listItem.appendChild(dragHandle);
              listItem.appendChild(label);
              listItem.appendChild(removeButton);

              listItem.addEventListener("dragstart", (e) => {
                draggedItem = listItem;
                draggedIndex = index;
                listItem.classList.add("dragging");
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/html", listItem.outerHTML);
              });

              listItem.addEventListener("dragend", () => {
                listItem.classList.remove("dragging");
                draggedItem = null;
                draggedIndex = null;
                clearDropIndicators();
              });

              listItem.addEventListener("dragover", (e) => {
                if (draggedItem && draggedItem !== listItem) {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = "move";

                  let rect = listItem.getBoundingClientRect();
                  let midpoint = rect.top + rect.height / 2;
                  let newDropPosition = e.clientY < midpoint ? "top" : "bottom";

                  if (dropTarget !== listItem || dropPosition !== newDropPosition) {
                    clearDropIndicators();
                    dropTarget = listItem;
                    dropPosition = newDropPosition;
                    showDropIndicator(listItem, newDropPosition);
                  }
                }
              });

              listItem.addEventListener("dragleave", (e) => {
                if (!listItem.contains(e.relatedTarget)) {
                  clearDropIndicators();
                }
              });

              listItem.addEventListener("drop", (e) => {
                e.preventDefault();
                if (draggedItem && draggedItem !== listItem) {
                  let targetIndex = parseInt(listItem.dataset.index);
                  let newIndex = targetIndex;

                  if (dropPosition === "bottom") {
                    newIndex = targetIndex + 1;
                  }

                  if (draggedIndex < newIndex) {
                    newIndex--;
                  }

                  reorderItems(draggedIndex, newIndex);
                }
                clearDropIndicators();
              });

              container.appendChild(listItem);
            });

            el.appendChild(container);

            let addInput = document.createElement("input");
            addInput.type = "text";
            addInput.className = "add-input";
            addInput.placeholder = "Add new item...";
            addInput.onkeydown = (e) => {
              if (e.key === "Enter" && addInput.value.trim()) {
                e.preventDefault();
                addItem(addInput.value.trim());
                addInput.value = "";
                addInput.focus();
              }
            };

            el.appendChild(addInput);
          }

          function addItem(text) {
            model.set("value", [...model.get("value"), text]);
            model.save_changes();
          }

          function removeItem(index) {
            model.set("value",  model.get("value").toSpliced(index, 1));
            model.save_changes();
          }

          function showDropIndicator(element, position) {
            let indicator = document.createElement("div");
            indicator.className = "drop-indicator";
            indicator.style.position = "absolute";
            indicator.style.left = "0";
            indicator.style.right = "0";
            indicator.style.height = "2px";
            indicator.style.backgroundColor = "#0066cc";
            indicator.style.zIndex = "1000";

            if (position === "top") {
              indicator.style.top = "-1px";
            } else {
              indicator.style.bottom = "-1px";
            }

            element.style.position = "relative";
            element.appendChild(indicator);
          }

          function clearDropIndicators() {
            el.querySelectorAll(".drop-indicator").forEach(indicator => {
              indicator.remove();
            });
            dropTarget = null;
            dropPosition = null;
          }

          function reorderItems(fromIndex, toIndex) {
            let items = [...model.get("value")];
            let [movedItem] = items.splice(fromIndex, 1);
            items.splice(toIndex, 0, movedItem);
            model.set("value", items);
            model.save_changes();
          }

          renderList();
          model.on("change:value", renderList);
        }

        export default { render };
        """
        _css = """
        .draggable-list-widget {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
          max-width: 100%;

          .list-container {
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          }      
          .list-item {
            position: relative;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            background: white;  
            border-bottom: 1px solid #e1e5e9;
            transition: background-color 0.15s ease, opacity 0.15s ease;
            cursor: grab;
          }      
          .list-item:last-child {
            border-bottom: none;
          }      
          .list-item:hover {
            background-color: #f8f9fa;
          }      
          .list-item:hover .remove-button {
            opacity: 1;
          }      
          .list-item.dragging {
            opacity: 0.5;
            cursor: grabbing;
          }      
          .drag-handle {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border: none;
            background: transparent;
            cursor: grab;
            color: #6b778c;
            flex-shrink: 0;
          }      
          .drag-handle:active {
            cursor: grabbing;
          }      
          .drag-handle svg {
            fill: currentColor;
          }      
          .item-label {
            flex: 1;
            color: #172b4d;
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }      
          .remove-button {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 3px;
            color: #6b778c;
            flex-shrink: 0;
            opacity: 0;
            transition: opacity 0.15s ease, background-color 0.15s ease;
          }      
          .remove-button:hover {
            background-color: #e4e6ea;
            color: #42526e;
          }      
          .add-input {
            width: 100%;
            padding: 8px 10px;
            margin-top: 8px;
            border: none;
            font-size: 14px;
            outline: none;
            background: transparent;
            color: #6b778c;
          }      
          .add-input:focus {
            background: #f8f9fa;
            color: #172b4d;
            border-radius: 3px;
          }      
          .drop-indicator {
            background-color: #0052cc !important;
            border-radius: 1px;
          }
        }
        """
        value = traitlets.List(traitlets.Unicode()).tag(sync=True)

        def __init__(self, value: typing.Sequence[str]) -> None:
            super().__init__(value=value)

    return (Sortable,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
