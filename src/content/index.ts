// ---------------- 유튜브 STT + 자막 (AudioWorklet 안정 버전) ----------------
let ws: WebSocket | null = null;
let audioCtx: AudioContext | null = null;
let sttNode: AudioWorkletNode | null = null;
let pingTimer: number | null = null;
let lastRTT: number | null = null;

async function startAudioCapture(lang: string = "auto") {
  const video = document.querySelector<HTMLVideoElement>("video");
  if (!video) return console.error("유튜브 비디오를 찾을 수 없습니다.");

  stopAudioCapture();

  // WebSocket 연결
  ws = new WebSocket("wss://livesubtitle.seokmin100.com");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    console.log("WebSocket 연결 성공");

    // 🔥 RTT 측정용 ping
    pingTimer = window.setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(`ping:${performance.now()}`);
      }
    }, 1000); // 1초 주기
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string" && e.data.startsWith("ping:")) {
      const sent = Number(e.data.slice(5));
      lastRTT = performance.now() - sent;
      return;
    }

    displaySubtitle(e.data as string);
  };

  ws.onclose = () => console.log("WebSocket 종료");
  ws.onerror = (e) => console.error("WebSocket 에러", e);

  // AudioContext 생성
  audioCtx = new AudioContext({ sampleRate: 16000 });

  try {
    // AudioWorklet 외부 파일 로드 (확장 환경 안전)
    await audioCtx.audioWorklet.addModule(chrome.runtime.getURL("stt-processor.js"));
  } catch (err) {
    console.error("AudioWorklet 로드 실패:", err);
    return;
  }

  // captureStream() TS 안전 캐스팅
  const stream: MediaStream = (video as unknown as { captureStream: () => MediaStream }).captureStream();
  const source = audioCtx.createMediaStreamSource(stream);

  // AudioWorkletNode 생성
  sttNode = new AudioWorkletNode(audioCtx, "stt-processor");
  
  let pcmBuffer: ArrayBuffer[] = [];
  let pcmLength = 0;
  const TARGET_SAMPLES = 16000 * 2.5 // 2.5초

  sttNode.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const { audio, rms } = e.data;
    if (!audio || !(audio instanceof ArrayBuffer)) return;

    // 디버그용
    console.log(`RMS: ${rms.toFixed(4)} | RTT: ${lastRTT ? lastRTT.toFixed(1) : "…"} ms`);

    pcmBuffer.push(audio);
    pcmLength += audio.byteLength / 2; // int16 = 2 bytes

    if (pcmLength >= TARGET_SAMPLES) {
      const merged = new Int16Array(pcmLength);
      let offset = 0;

      for (const buf of pcmBuffer) {
        const arr = new Int16Array(buf);
        merged.set(arr, offset);
        offset += arr.length;
      }

      ws.send(merged.buffer);

      pcmBuffer = [];
      pcmLength = 0;
    }
  };

  source.connect(sttNode);
  // source.connect(audioCtx.destination); // 원본 소리 재생
  // 필요 시 sttNode.connect(audioCtx.destination);
}

function stopAudioCapture() {
  if (sttNode) { sttNode.disconnect(); sttNode = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws) { ws.close(); ws = null; }

  if (pingTimer) {
    clearInterval(pingTimer);
    pingTimer = null;
  }

  clearSubtitle();
}

// ---------------- DOM 유틸 ----------------
function displaySubtitle(text: string) {
  const player = document.querySelector<HTMLDivElement>(".html5-video-player");
  if (!player) return;

  let subtitleDiv = document.getElementById("my-extension-subtitle") as HTMLDivElement | null;
  if (!subtitleDiv) {
    subtitleDiv = document.createElement("div") as HTMLDivElement;
    subtitleDiv.id = "my-extension-subtitle";

    subtitleDiv.style.position = "absolute";
    subtitleDiv.style.bottom = "80px";
    subtitleDiv.style.left = "50%";
    subtitleDiv.style.transform = "translateX(-50%)";

    subtitleDiv.style.background = "rgba(0,0,0,0.7)";
    subtitleDiv.style.color = "#fff";
    subtitleDiv.style.padding = "4px 8px";
    subtitleDiv.style.borderRadius = "4px";
    subtitleDiv.style.fontSize = "14px";
    subtitleDiv.style.fontFamily = "Roboto, Arial, sans-serif";
    subtitleDiv.style.zIndex = "9999";

    player.appendChild(subtitleDiv);

    // 드래그 + 크기조절 적용
    makeSubtitleDraggableAndResizable(subtitleDiv);
  }

  subtitleDiv.textContent = text;
}

function clearSubtitle() {
  const subtitleDiv = document.getElementById("my-extension-subtitle");
  if (subtitleDiv) subtitleDiv.remove();
}

// ---------------- 드래그 & 크기조절 ----------------
function makeSubtitleDraggableAndResizable(subtitleDiv: HTMLDivElement) {
  let isDragging = false;
  let isResizing = false;
  let startX = 0, startY = 0;
  let startWidth = 0, startHeight = 0;
  let origX = 0, origY = 0;

  let resizeEdges = { top: false, right: false, bottom: false, left: false };
  const edgeThreshold = 6;

  subtitleDiv.addEventListener("mousedown", (e) => {
    const parent = subtitleDiv.parentElement;
    if (!parent) return; 
    const parentRect = parent.getBoundingClientRect();

    const rect = subtitleDiv.getBoundingClientRect();
    startX = e.clientX;
    startY = e.clientY;

    // 부모 기준 좌표
    origX = rect.left - parentRect.left;
    origY = rect.top - parentRect.top;

    // 각 테두리 근처 체크
    resizeEdges.top = startY >= rect.top && startY <= rect.top + edgeThreshold;
    resizeEdges.bottom = startY >= rect.bottom - edgeThreshold && startY <= rect.bottom;
    resizeEdges.left = startX >= rect.left && startX <= rect.left + edgeThreshold;
    resizeEdges.right = startX >= rect.right - edgeThreshold && startX <= rect.right;

    if (resizeEdges.top || resizeEdges.bottom || resizeEdges.left || resizeEdges.right) {
      isResizing = true;
      startWidth = rect.width;
      startHeight = rect.height;
    } else {
      isDragging = true;
    }

    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    const parent = subtitleDiv.parentElement;
    if (!parent) return; // 부모가 없으면 중단
    const parentRect = parent.getBoundingClientRect();

    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    if (isDragging) {
      subtitleDiv.style.left = `${origX + dx}px`;
      subtitleDiv.style.top = `${origY + dy}px`;
      subtitleDiv.style.bottom = "auto";
      subtitleDiv.style.transform = "none";
    } else if (isResizing) {
      let newWidth = startWidth;
      let newHeight = startHeight;
      let newLeft = origX;
      let newTop = origY;

      if (resizeEdges.right) newWidth = startWidth + dx;
      if (resizeEdges.left) {
        newWidth = startWidth - dx;
        newLeft = origX + dx;
      }
      if (resizeEdges.bottom) newHeight = startHeight + dy;
      if (resizeEdges.top) {
        newHeight = startHeight - dy;
        newTop = origY + dy;
      }

      // 최소 크기 유지
      newWidth = Math.max(50, newWidth);
      newHeight = Math.max(20, newHeight);

      subtitleDiv.style.width = `${newWidth}px`;
      subtitleDiv.style.height = `${newHeight}px`;
      subtitleDiv.style.left = `${newLeft}px`;
      subtitleDiv.style.top = `${newTop}px`;
      subtitleDiv.style.bottom = "auto";
      subtitleDiv.style.transform = "none";
    }
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
    isResizing = false;
  });

  // 커서 변경
  subtitleDiv.addEventListener("mousemove", (e) => {
    const rect = subtitleDiv.getBoundingClientRect();
    const top = e.clientY >= rect.top && e.clientY <= rect.top + edgeThreshold;
    const bottom = e.clientY >= rect.bottom - edgeThreshold && e.clientY <= rect.bottom;
    const left = e.clientX >= rect.left && e.clientX <= rect.left + edgeThreshold;
    const right = e.clientX >= rect.right - edgeThreshold && e.clientX <= rect.right;

    if ((top && left) || (bottom && right)) {
      subtitleDiv.style.cursor = "nwse-resize";
    } else if ((top && right) || (bottom && left)) {
      subtitleDiv.style.cursor = "nesw-resize";
    } else if (top || bottom) {
      subtitleDiv.style.cursor = "ns-resize";
    } else if (left || right) {
      subtitleDiv.style.cursor = "ew-resize";
    } else {
      subtitleDiv.style.cursor = "move";
    }
  });
}

// ---------------- 유튜브 버튼 & 팝업 ----------------
let popupVisible = false;
let stopSTT: (() => void) | null = null;

function injectButton() {
  const controls = document.querySelector<HTMLDivElement>(".ytp-right-controls");
  const playerControls = document.querySelector<HTMLDivElement>(".ytp-chrome-bottom");
  if (!controls || !playerControls) return;

  // 기존 버튼/팝업 제거
  const oldBtn = document.getElementById("my-extension-btn");
  const oldPopup = document.getElementById("my-extension-popup");
  if (oldBtn) oldBtn.remove();
  if (oldPopup) oldPopup.remove();
  clearSubtitle();
  if (stopSTT) stopSTT();
  stopSTT = null;
  popupVisible = false;

  const btn = document.createElement("button");
  btn.id = "my-extension-btn";
  btn.className = "ytp-button";
  btn.title = "LiveSubtitle Settings";
  btn.style.width = "48px";
  btn.style.height = "40px";
  btn.style.backgroundImage = `url(${chrome.runtime.getURL("img/icon.png")})`;
  btn.style.backgroundRepeat = "no-repeat";
  btn.style.backgroundPosition = "center";
  btn.style.backgroundSize = "22px 22px";

  const popup = document.createElement("div");
  popup.id = "my-extension-popup";
  popup.style.position = "absolute";
  popup.style.bottom = `${playerControls.offsetHeight + 8}px`;
  popup.style.right = "10px";
  popup.style.width = "260px";
  popup.style.background = "#212121";
  popup.style.color = "#fff";
  popup.style.fontFamily = "Roboto, Arial, sans-serif";
  popup.style.borderRadius = "8px";
  popup.style.boxShadow = "0 4px 12px rgba(0,0,0,0.5)";
  popup.style.padding = "12px";
  popup.style.zIndex = "9999";
  popup.style.opacity = "0";
  popup.style.visibility = "hidden";
  popup.style.transition = "opacity 0.3s ease, visibility 0.3s ease";

// 팝업 HTML 수정 부분
  popup.innerHTML = `
    <div style="font-weight:500; margin-bottom:10px;">Live Subtitle Settings</div>
    
    <div class="yt-setting-item" style="margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
      <span>자막 켜기</span>
      <label class="yt-switch">
        <input type="checkbox" id="subtitle-toggle" />
        <span class="slider"></span>
      </label>
    </div>

    <div class="yt-setting-item" style="margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
      <span>언어 선택</span>
      <select id="subtitle-lang" style="background:#212121; color:#fff; border:1px solid #303030; border-radius:4px; padding:2px 4px;">
        <option value="auto">자동</option>
        <option value="en">English</option>
        <option value="ko">한국어</option>
        <option value="ja">日本語</option>
      </select>
    </div>

    <div class="yt-setting-item" style="margin-bottom:8px; display:flex; flex-direction:column;">
      <label for="subtitle-font-size" style="margin-bottom:4px;">폰트 크기</label>
      <input type="range" id="subtitle-font-size" min="10" max="50" value="14" />
    </div>
  `;

  // 슬라이더 이벤트 적용
  const fontSizeSlider = popup.querySelector<HTMLInputElement>("#subtitle-font-size");
  fontSizeSlider?.addEventListener("input", () => {
    const subtitleDiv = document.getElementById("my-extension-subtitle");
    if (subtitleDiv) {
      subtitleDiv.style.fontSize = `${fontSizeSlider.value}px`;
    }
  });

  playerControls.parentElement?.appendChild(popup);
  controls.prepend(btn);

  const subtitleToggle = popup.querySelector<HTMLInputElement>("#subtitle-toggle");
  const subtitleLang = popup.querySelector<HTMLSelectElement>("#subtitle-lang");

  btn.addEventListener("click", () => {
    popupVisible = !popupVisible;
    popup.style.opacity = popupVisible ? "1" : "0";
    popup.style.visibility = popupVisible ? "visible" : "hidden";

    if (popupVisible) {
      playerControls.style.opacity = "1";
      playerControls.style.visibility = "visible";
    } else {
      playerControls.style.opacity = "";
      playerControls.style.visibility = "";
    }
  });

  subtitleToggle?.addEventListener("change", () => {
    if (subtitleToggle.checked) startAudioCapture();
    else stopAudioCapture();
  });

  // 재생바 변화 감지
  const barObserver = new MutationObserver(() => {
    const barStyle = window.getComputedStyle(playerControls);
    const barVisible = barStyle.opacity !== "0" && !playerControls.classList.contains("ytp-autohide");
    popup.style.opacity = popupVisible && barVisible ? "1" : "0";
    popup.style.visibility = popupVisible && barVisible ? "visible" : "hidden";
  });
  barObserver.observe(playerControls, { attributes: true, attributeFilter: ["style", "class"] });

  // 재생바 다른 버튼 클릭 시 팝업 닫기
  playerControls.querySelectorAll<HTMLButtonElement>(".ytp-button").forEach(b => {
    if (b.id === "my-extension-btn") return;
    b.addEventListener("click", () => {
      if (popupVisible) {
        popupVisible = false;
        popup.style.opacity = "0";
        popup.style.visibility = "hidden";
      }
    });
  });

  // 스위치 CSS
  if (!document.getElementById("yt-switch-style")) {
    const style = document.createElement("style");
    style.id = "yt-switch-style";
    style.textContent = `
      .yt-switch { position: relative; display: inline-block; width: 36px; height: 20px; }
      .yt-switch input { display: none; }
      .yt-switch .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
        background-color: #3f3f3f; border-radius: 20px; transition: 0.2s; }
      .yt-switch .slider::before { position: absolute; content: ""; height: 16px; width: 16px; left: 2px; bottom: 2px;
        background-color: white; border-radius: 50%; transition: 0.2s; }
      .yt-switch input:checked + .slider { background-color: #ff0000; }
      .yt-switch input:checked + .slider::before { transform: translateX(16px); }
    `;
    document.head.appendChild(style);
  }

  // 슬라이더 CSS
  if (!document.getElementById("subtitle-slider-style")) {
    const style = document.createElement("style");
    style.id = "subtitle-slider-style";
    style.textContent = `
      #subtitle-font-size {
        -webkit-appearance: none;
        width: 100%;
        height: 6px;
        background: #555; /* 트랙 기본 회색 */
        border-radius: 3px;
        outline: none;
      }
      #subtitle-font-size::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 20px;
        height: 20px;
        background-color: #fff; /* 버튼 흰색 */
        border-radius: 50%;
      }
    `;
    document.head.appendChild(style);

    // 빨간색 채움 업데이트
    const slider = document.getElementById("subtitle-font-size") as HTMLInputElement | null;
    if (slider) {
      const updateSliderFill = () => {
        const percent = ((Number(slider.value) - Number(slider.min)) / (Number(slider.max) - Number(slider.min))) * 100;
        slider.style.background = `linear-gradient(to right, #ff0000 0%, #ff0000 ${percent}%, #555 ${percent}%, #555 100%)`;
      };
      slider.addEventListener("input", updateSliderFill);
      updateSliderFill(); // 초기값 적용
    }
  }
}

// ---------------- SPA / URL 변경 대응 ----------------
let lastUrl = location.href;
setInterval(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    injectButton();
  }
}, 500);

injectButton();
