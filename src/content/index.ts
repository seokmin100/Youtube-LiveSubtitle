// 유튜브 우측 컨트롤에 커스텀 버튼 추가
function injectButton() {
  const controls = document.querySelector<HTMLDivElement>(
    ".ytp-right-controls"
  );
  if (!controls) return;

  // 중복 삽입 방지
  if (document.getElementById("my-extension-btn")) return;

  const btn = document.createElement("button");
  btn.id = "my-extension-btn";
  btn.className = "ytp-button";
  btn.title = "My Extension Button";

  // 버튼 스타일
  btn.style.width = "48px";
  btn.style.height = "40px";
  btn.style.backgroundImage = `url(${chrome.runtime.getURL(
    "img/icon.png"
  )})`;
  btn.style.backgroundRepeat = "no-repeat";
  btn.style.backgroundPosition = "center";
  btn.style.backgroundSize = "22px 22px";

  // 클릭 이벤트
  btn.addEventListener("click", () => {
    alert("확장프로그램 버튼 클릭!");
  });

  // 우측 컨트롤 맨 앞에 추가
  controls.prepend(btn);
}

// YouTube는 SPA라서 DOM 변경 감시 필요
const observer = new MutationObserver(() => {
  injectButton();
});

observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
});

// 혹시 초기 로딩 놓칠까봐 한 번 즉시 실행
injectButton();
