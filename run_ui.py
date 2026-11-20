"""Local web UI. Zero external assets, zero network, starts in <1 s.

    python run_ui.py
    python run_ui.py --port 8000 --no-browser
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.engine import TuringBot  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Happy · Тьюринг Арена</title>
<style>
  :root{
    --bg:#0a090c; --card:#141218; --card2:#1c1922; --line:#2a2632;
    --fg:#f4f1ea; --mut:#8d8798; --acc:#c9ff4f;
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0;color:var(--fg);background:
      radial-gradient(1200px 600px at 50% -10%, #201a2b 0%, transparent 60%),
      radial-gradient(800px 400px at 100% 100%, #14201a 0%, transparent 55%),
      var(--bg);
    display:flex;flex-direction:column;
  }
  header{
    display:flex;align-items:center;gap:14px;
    padding:18px 26px;border-bottom:1px solid var(--line);
    backdrop-filter:blur(8px);
  }
  .dot{width:9px;height:9px;border-radius:50%;background:var(--acc);box-shadow:0 0 14px var(--acc)}
  .title{font-weight:650;letter-spacing:.2px}
  .tag{
    font:11px ui-monospace,SFMono-Regular,Menlo,monospace;
    color:var(--mut);border:1px solid var(--line);border-radius:999px;
    padding:4px 9px;
  }
  .spacer{flex:1}
  main{flex:1;overflow-y:auto;padding:26px 18px 6px;scroll-behavior:smooth}
  .wrap{max-width:820px;margin:0 auto}
  .row{display:flex;margin-bottom:16px;animation:in .22s ease-out}
  .row.me{justify-content:flex-end}
  @keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .bubble{
    max-width:78%;padding:12px 15px;border-radius:16px;line-height:1.45;
    font-size:17px;white-space:pre-wrap;word-break:break-word;
  }
  .me .bubble{background:#262231;border:1px solid #332d40;border-bottom-right-radius:5px}
  .her .bubble{background:var(--card);border:1px solid var(--line);border-bottom-left-radius:5px}
  .meta{
    font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);
    margin:6px 2px 0;
  }
  .typing{display:inline-flex;gap:5px;align-items:center;height:10px}
  .typing i{width:6px;height:6px;border-radius:50%;background:var(--mut);animation:b 1.1s infinite}
  .typing i:nth-child(2){animation-delay:.15s}
  .typing i:nth-child(3){animation-delay:.3s}
  @keyframes b{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)}}
  footer{padding:14px 18px 22px}
  .bar{max-width:820px;margin:0 auto;display:flex;gap:10px;align-items:center}
  input[type=text]{
    flex:1;background:var(--card);color:var(--fg);border:1px solid var(--line);
    border-radius:14px;padding:15px 16px;font-size:17px;outline:none;
  }
  input[type=text]:focus{border-color:#4b4359;box-shadow:0 0 0 3px #4b435922}
  button{
    background:var(--acc);color:#10120a;border:0;border-radius:14px;
    padding:15px 20px;font-size:15px;font-weight:650;cursor:pointer;
  }
  button:active{transform:translateY(1px)}
  .opts{
    max-width:820px;margin:10px auto 0;display:flex;gap:16px;align-items:center;
    color:var(--mut);font-size:12px;
  }
  .opts label{display:flex;gap:6px;align-items:center;cursor:pointer}
  kbd{font:11px ui-monospace;border:1px solid var(--line);border-radius:5px;padding:1px 5px;color:var(--mut)}
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <span class="title">Happy</span>
  <span class="tag">Тьюринг Арена · локально · без сети</span>
  <span class="spacer"></span>
  <span class="tag" id="round">раунд 0</span>
</header>

<main id="log"><div class="wrap" id="wrap">
  <div class="row her"><div class="bubble">привет. задавай вопрос, отвечу)<div class="meta">готов</div></div></div>
</div></main>

<footer>
  <div class="bar">
    <input id="q" type="text" placeholder="Вопрос из зала…" autocomplete="off" autofocus>
    <button id="send">Ответить</button>
  </div>
  <div class="opts">
    <label><input type="checkbox" id="pace" checked> человеческая пауза</label>
    <span>Enter — отправить · <kbd>/reset</kbd> — очистить</span>
  </div>
</footer>

<script>
const log = document.getElementById('wrap');
const input = document.getElementById('q');
const pace = document.getElementById('pace');
const roundEl = document.getElementById('round');
let round = 0;

function el(html){const d=document.createElement('div');d.innerHTML=html;return d.firstElementChild;}
function scroll(){document.querySelector('main').scrollTop=1e9;}

function addMe(text){
  const row = el('<div class="row me"><div class="bubble"></div></div>');
  row.querySelector('.bubble').textContent = text;
  log.appendChild(row); scroll();
}
function addTyping(){
  const row = el('<div class="row her" id="typing"><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div></div>');
  log.appendChild(row); scroll(); return row;
}
function addHer(text, meta){
  const row = el('<div class="row her"><div class="bubble"><span class="body"></span><div class="meta"></div></div></div>');
  row.querySelector('.body').textContent = text;
  row.querySelector('.meta').textContent = meta;
  log.appendChild(row); scroll();
}

async function ask(){
  const text = input.value.trim();
  if(!text) return;
  input.value = '';
  if(text === '/reset'){
    await fetch('/reset',{method:'POST'});
    log.innerHTML=''; round=0; roundEl.textContent='раунд 0';
    addHer('ок, всё забыла) давай заново', 'сброс');
    return;
  }
  addMe(text);
  const typing = addTyping();
  const t0 = performance.now();
  let data;
  try{
    const r = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q:text, pace:pace.checked})});
    data = await r.json();
  }catch(e){
    typing.remove(); addHer('ой, что-то зависло. повтори?', 'ошибка'); return;
  }
  const wait = (pace.checked ? (data.typing_delay||0)*1000 : 0);
  setTimeout(()=>{
    typing.remove();
    const total = (performance.now()-t0)/1000;
    const meta = data.latency_ms.toFixed(0)+' мс · '+data.source+' · '+total.toFixed(1)+' с всего';
    addHer(data.text, meta);
    round++; roundEl.textContent='раунд '+round;
  }, Math.max(0, wait));
}
document.getElementById('send').onclick = ask;
input.addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); ask(); }});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    bot: TuringBot
    lock = threading.Lock()

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/reset":
            with self.lock:
                self.bot.reset()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path != "/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            payload = json.loads(raw or b"{}")
            question = str(payload.get("q", ""))
            pacing = bool(payload.get("pace", False))
            with self.lock:
                answer = self.bot.answer(question, pacing=pacing)
            body = json.dumps(
                {
                    "text": answer.text,
                    "latency_ms": answer.latency_ms,
                    "typing_delay": answer.typing_delay,
                    "source": answer.source,
                    "confidence": answer.confidence,
                    "topic": answer.topic,
                    "intent": answer.intent,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except Exception as exc:  # never leave the operator with a blank screen
            body = json.dumps({"text": "ой, глюк. повтори?", "latency_ms": 0,
                               "typing_delay": 1.0, "source": "error",
                               "error": repr(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")

    def log_message(self, *args) -> None:  # silence per-request logs
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    bot = TuringBot(seed=args.seed)
    Handler.bot = bot
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Happy · локальный режим · {url}")
    print(f"корпус: {len(bot.corpus)} записей · офлайн, сеть не используется")
    print(f"dense: {'ON' if bot.dense is not None else 'OFF (BM25 only)'}")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nстоп")


if __name__ == "__main__":
    main()
