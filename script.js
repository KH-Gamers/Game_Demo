let eraseMode = false;
let cachedImageEmbedding = null;
let modelSession = null;
let penSize = 2;
let eraserSize = 10;
const historyList = [];
const imageList = ["6.png"]; // 로딩할 이미지 목록
const undoStack = [];

function setupCanvas(canvas) {
  function resizeCanvas() {
    const size = Math.min(window.innerWidth, window.innerHeight) * 0.5;  // 이 비율 조절 가능
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const ctx = canvas.getContext('2d');
  let drawing = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener('mousedown', (event) => {
    saveCanvasState(canvas);
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = event.clientX - rect.left;
    lastY = event.clientY - rect.top;
  });

  canvas.addEventListener('mouseup', () => drawing = false);
  canvas.addEventListener('mouseout', () => drawing = false);

  canvas.addEventListener('mousemove', (event) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.strokeStyle = eraseMode ? 'white' : 'black';
    ctx.lineWidth = (eraseMode ? eraserSize : penSize) * 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;
  });
}

function getCanvasRGBData(canvas) {
  const offscreen = document.createElement('canvas');
  offscreen.width = 224;
  offscreen.height = 224;

  const ctx = offscreen.getContext('2d');
  ctx.drawImage(canvas, 0, 0, offscreen.width, offscreen.height);

  const imageData = ctx.getImageData(0, 0, 224, 224);
  const data = imageData.data;
  const rgbData = [];

  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < data.length; i += 4) {
      rgbData.push(data[i + c] / 255);
    }
  }
  return rgbData;
}

function cosineSimilarity(vec1, vec2) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vec1.length; i++) {
    dot += vec1[i] * vec2[i];
    normA += vec1[i] ** 2;
    normB += vec2[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function runViT(rgbData) {
  const inputTensor = new Float32Array(rgbData);
  const inputShape = [1, 3, 224, 224];
  const feeds = { pixel_values: new ort.Tensor('float32', inputTensor, inputShape) };
  const results = await modelSession.run(feeds);
  const output = results["last_hidden_state"];
  return output.data.slice(0, 768); // CLS token
}

async function sendToModel() {
  const canvas = document.getElementById('canvas1');
  const canvasRGB = getCanvasRGBData(canvas);
  const vec1 = await runViT(canvasRGB);

  if (!cachedImageEmbedding) {
    alert("Error: reference image embedding not ready.");
    return;
  }

  const similarity = cosineSimilarity(vec1, cachedImageEmbedding);
  document.getElementById('similarity-result').innerText = similarity.toFixed(4);

  const imageDataUrl = canvas.toDataURL();
  addToHistory(similarity, imageDataUrl);

  if (similarity >= 0.7) {
    document.getElementById('overlay').style.display = 'flex';
  }
}

async function loadRandomImage(imageList) {
  const randomImage = imageList[Math.floor(Math.random() * imageList.length)];
  document.getElementById("loaded-image-name").innerText = randomImage;

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = async () => {
    const offscreen = document.createElement("canvas");
    offscreen.width = 224;
    offscreen.height = 224;
    const ctx = offscreen.getContext("2d");
    ctx.drawImage(img, 0, 0, 224, 224);
    const rgbData = getCanvasRGBData(offscreen);
    cachedImageEmbedding = await runViT(rgbData);
    console.log("Reference image embedding ready.");
  };
  img.src = `problems/${randomImage}`;
}

function renderHistory() {
  const listContainer = document.getElementById('similarity-history');
  listContainer.innerHTML = '';

  const sortedHistory = [...historyList].sort((a, b) => b.similarity - a.similarity);
  sortedHistory.forEach((entry) => {
    const item = document.createElement('li');
    const label = document.createElement('div');
    label.textContent = `${entry.similarity.toFixed(4)}`;
    label.style.fontWeight = 'bold';

    const contentWrapper = document.createElement('div');
    contentWrapper.style.marginTop = '0.5rem';

    let expanded = false;

    label.addEventListener('click', () => {
      contentWrapper.innerHTML = '';
      if (!expanded) {
        const img = new Image();
        img.src = entry.imageDataUrl;
        img.style.width = '150px';
        img.style.border = '1px solid #ccc';
        img.style.borderRadius = '8px';
        img.style.display = 'block';
        img.style.marginBottom = '0.5rem';

        const loadButton = document.createElement('button');
        loadButton.textContent = '캔버스로 불러오기';
        loadButton.addEventListener('click', () => {
          const canvas = document.getElementById('canvas1');
          const ctx = canvas.getContext('2d');
          const imgToLoad = new Image();
          imgToLoad.onload = () => {
            ctx.drawImage(imgToLoad, 0, 0, canvas.width, canvas.height);
          };
          imgToLoad.src = entry.imageDataUrl;
        });

        const closeButton = document.createElement('button');
        closeButton.textContent = '닫기';
        closeButton.addEventListener('click', () => {
          contentWrapper.innerHTML = '';
          label.style.color = 'gray';
          expanded = false;
        });

        contentWrapper.appendChild(img);
        contentWrapper.appendChild(loadButton);
        contentWrapper.appendChild(closeButton);

        label.style.color = 'black';
        expanded = true;
      } else {
        contentWrapper.innerHTML = '';
        label.style.color = 'gray';
        expanded = false;
      }
    });

    item.appendChild(label);
    item.appendChild(contentWrapper);
    listContainer.appendChild(item);
  });
}

function addToHistory(similarity, imageDataUrl) {
  historyList.push({ similarity, imageDataUrl });
  renderHistory();
}

function saveCanvasState(canvas) {
  undoStack.push(canvas.toDataURL());
  if (undoStack.length > 10) undoStack.shift(); // 최대 10개까지 저장
}

function undoLastAction() {
  if (undoStack.length === 0) return;

  const canvas = document.getElementById('canvas1');
  const ctx = canvas.getContext('2d');
  const previousImage = new Image();
  previousImage.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(previousImage, 0, 0, canvas.width, canvas.height);
  };
  const dataUrl = undoStack.pop();
  previousImage.src = dataUrl;
}

document.getElementById('compare-button').addEventListener('click', sendToModel);
document.getElementById('toggle-mode-button').addEventListener('click', () => {
  eraseMode = !eraseMode;
  document.getElementById('toggle-mode-button').innerText = eraseMode ? 'Mode: Erase' : 'Mode: Draw';
});
document.getElementById('clear-canvas-button').addEventListener('click', () => {
  const canvas = document.getElementById('canvas1');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});
document.getElementById('pen-size-slider').addEventListener('input', (e) => {
  penSize = parseInt(e.target.value);
  document.getElementById('pen-size-label').innerText = penSize;
});
document.getElementById('eraser-size-slider').addEventListener('input', (e) => {
  eraserSize = parseInt(e.target.value);
  document.getElementById('eraser-size-label').innerText = eraserSize;
});
document.getElementById('close-overlay-button').addEventListener('click', () => {
  document.getElementById('overlay').style.display = 'none';
});
document.getElementById('undo-button').addEventListener('click', undoLastAction);

window.addEventListener('DOMContentLoaded', async () => {
  const canvas = document.getElementById('canvas1');
  setupCanvas(canvas);
  modelSession = await ort.InferenceSession.create('model_quantized.onnx');
  await loadRandomImage(imageList);
});


