#date: 2025-12-02T17:03:20Z
#url: https://api.github.com/gists/78a039652f747b065f576cc906ec407c
#owner: https://api.github.com/users/aont

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Color Picker Page</title>
  <style>
    body {
      font-family: sans-serif;
      padding: 20px;
    }
    .preview {
      width: 200px;
      height: 200px;
      border: 1px solid #ccc;
      margin-top: 20px;
    }
    #history li {
      cursor: pointer;
      list-style: none;
      margin: 4px 0;
    }
  </style>
  
  <script>
window.addEventListener("load", (ev) => {
    const picker = document.getElementById('colorPicker');
    const preview = document.getElementById('preview');
    const history = document.getElementById('history');
    const addToHistoryBtn = document.getElementById('addToHistory');

    // 現在選択中の色をプレビューに反映
    function updatePreview() {
        preview.style.backgroundColor = picker.value;
    }

    // 履歴に現在の色を追加
    function addCurrentColorToHistory() {
        const color = picker.value;

        const li = document.createElement('li');
        li.textContent = color;
        li.style.color = color;
        li.dataset.color = color; // 後で参照しやすいように data 属性に保持
        history.appendChild(li);
    }

    // カラーピッカー変更時にプレビューだけ更新（履歴には追加しない）
    picker.addEventListener('input', updatePreview);

    // 履歴に追加ボタンクリックで履歴に追加
    addToHistoryBtn.addEventListener('click', addCurrentColorToHistory);

    // 履歴の色をクリックしたら、その色を現在の色に反映
    history.addEventListener('click', (event) => {
        const target = event.target;
        if (target.tagName.toLowerCase() === 'li') {
        const color = target.dataset.color;
        picker.value = color;
        updatePreview();
        }
    });

    // 初期表示
    updatePreview();
});
  </script>
</head>
<body>
  <h1>Select a Color</h1>
  <input type="color" id="colorPicker" value="#ff0000" />
  <button id="addToHistory">Add</button>

  <div class="preview" id="preview"></div>

  <h2>Selection History</h2>
  <ul id="history"></ul>

</body>
</html>
