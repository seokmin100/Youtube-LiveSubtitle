import { WebSocketServer } from "ws";

const PORT = 3000;

const wss = new WebSocketServer({ port: PORT });

console.log(`✅ WebSocket STT 서버 실행됨 : ws://localhost:${PORT}`);

wss.on("connection", (ws, req) => {
  console.log("🔗 클라이언트 연결됨");

  ws.on("message", (data) => {
    // data는 ArrayBuffer (PCM 16bit)
    if (data instanceof Buffer) {
      console.log("🎧 오디오 수신:", data.length, "bytes");

      // 테스트용: 더미 자막 반환
      ws.send("🎤 음성 수신 중...");
    }
  });

  ws.on("close", () => {
    console.log("❌ 클라이언트 연결 종료");
  });

  ws.on("error", (err) => {
    console.error("⚠️ WS 에러:", err.message);
  });
});
