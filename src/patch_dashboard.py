#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_dashboard.py (v2)
────────────────────────────────────────────────────────────────
dashboard.html 에 다음을 자동 반영한다.
  1) 1/5/10분 미누적 시 화면에 "측정중입니다" 만 표시
  2) 음성 캘리브레이션: 클릭 → 팝업 → ①소음측정 ②테스트발화 + 실시간 레벨 막대
  3) 신호등 옆 "상세 원인 분석" 버튼 → 분석 탭(신호등 결정 단계 표시)

반드시 '원본' dashboard.html 에 적용하세요. (이전 패치본이면 .bak 으로 복원 후 실행)
원본은 dashboard.html.bak 으로 백업되며, 재실행해도 중복 반영되지 않는다.

사용:  python3 patch_dashboard.py [dashboard.html 경로]
────────────────────────────────────────────────────────────────
"""

import sys, os, shutil

PATH = sys.argv[1] if len(sys.argv) > 1 else "dashboard.html"

CALIB_MODAL = """<!-- 음성 캘리브레이션 안내 모달 -->
<div class="mo" id="calib-mo"><div class="mc" style="text-align:center;max-width:420px">
  <h2 style="font-size:1.05rem">&#127908; 음성 캘리브레이션</h2>
  <div id="calib-stage" style="font-size:1.3rem;font-weight:800;margin:14px 0 6px"></div>
  <div id="calib-guide" style="font-size:.9rem;color:var(--tx2);line-height:1.5;min-height:2.4em"></div>
  <div id="calib-count" style="font-size:2rem;font-weight:800;color:var(--state);margin:8px 0"></div>
  <div id="calib-phrase" style="font-size:1.15rem;font-weight:700;color:var(--green);margin:6px 0;min-height:1.4em"></div>
  <div id="calib-level-wrap" style="display:none;margin:10px auto 2px;max-width:300px">
    <div style="height:16px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;overflow:hidden">
      <div id="calib-level-bar" style="height:100%;width:0%;background:var(--green);transition:width .12s"></div>
    </div>
    <div style="font-size:.7rem;color:var(--tx3);margin-top:3px">실시간 입력 레벨</div>
  </div>
</div></div>

"""

OLD_VCALIB = """  /* ───────── 음성 캘리브레이션 ───────── */
  document.getElementById('vcalibBtn').addEventListener('click',function(){
    var msg=document.getElementById('vcmsg'),btn=this;btn.disabled=true;
    msg.style.color='#ffcc00';msg.textContent='음성 캘리브레이션 요청 중...';
    fetch('/api/speech-recalibrate',{method:'POST'}).then(function(r){return r.json();})
      .then(function(){msg.style.color='#66ff88';msg.textContent='완료 — 테스트 문장을 말해 보세요';setTimeout(function(){msg.textContent='';btn.disabled=false;},5000);})
      .catch(function(){msg.style.color='#ff6666';msg.textContent='요청 실패 (speech_agent 실행 중인지 확인)';btn.disabled=false;});
  });"""

NEW_VCALIB = """  /* ───────── 음성 캘리브레이션 (클릭 → 팝업 → 2단계 + 실시간 레벨) ───────── */
  var _calibTimer=null;
  function _calibUI(stage,guide,count,phrase){
    document.getElementById('calib-stage').textContent=stage||'';
    document.getElementById('calib-guide').textContent=guide||'';
    document.getElementById('calib-count').textContent=count||'';
    document.getElementById('calib-phrase').textContent=phrase||'';
  }
  function _calibLevel(show,lvl){
    var w=document.getElementById('calib-level-wrap'),b=document.getElementById('calib-level-bar');
    if(w)w.style.display=show?'block':'none';
    if(b&&show)b.style.width=Math.round(Math.min(1,lvl||0)*100)+'%';
  }
  document.getElementById('vcalibBtn').addEventListener('click',function(){
    var btn=this; btn.disabled=true;
    var seen='', pStart=0, pDur=0, deadline=Date.now()+10000;
    openModal('calib-mo');
    _calibUI('준비 중...','마이크 에이전트가 캘리브레이션을 시작하길 기다리는 중이에요.','','');
    _calibLevel(false,0);
    fetch('/api/speech-recalibrate',{method:'POST'}).catch(function(){});
    if(_calibTimer)clearInterval(_calibTimer);
    _calibTimer=setInterval(function(){
      fetch('/api/speech-calib-status').then(function(r){return r.json();}).then(function(d){
        var phase=d.phase||'idle';
        if(phase==='noise'||phase==='speak'){
          deadline=Date.now()+15000;
          if(phase!==seen){seen=phase;pStart=Date.now();pDur=(phase==='noise'?(d.noise_sec||1.5):(d.enroll_sec||4))*1000;}
          var remain=Math.max(0,Math.ceil((pDur-(Date.now()-pStart))/1000));
          if(phase==='noise')_calibUI('① 주변 소음 측정 중','지금은 조용히 해주세요. 주변 소음을 재고 있어요.',remain+'초','');
          else _calibUI('② 테스트 문장 말하기','아래 문장을 평소 톤으로 또렷이 말해주세요.',remain+'초','"'+(d.text||'결재 부탁드립니다')+'"');
          _calibLevel(true,d.level);
        }else if(phase==='done'&&(seen==='noise'||seen==='speak')){
          _calibUI('✅ 완료','캘리브레이션이 끝났어요.','','');_calibLevel(false,0);
          clearInterval(_calibTimer);_calibTimer=null;btn.disabled=false;
          setTimeout(function(){closeModal('calib-mo');},1500);
        }
        if(Date.now()>deadline){
          clearInterval(_calibTimer);_calibTimer=null;closeModal('calib-mo');btn.disabled=false;
          var m=document.getElementById('vcmsg');
          if(m){m.style.color='#ff6666';m.textContent='응답 없음 — mic_agent(발화) 실행 확인';}
        }
      }).catch(function(){
        if(Date.now()>deadline){clearInterval(_calibTimer);_calibTimer=null;closeModal('calib-mo');btn.disabled=false;}
      });
    },200);
  });"""

NAV_OLD = """      <span class="nl">결재</span>
    </button>
  </nav>"""
NAV_NEW = """      <span class="nl">결재</span>
    </button>
    <button class="nav" data-tab="analysis">
      <span class="ni"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5"/><path d="M4 19h16"/><rect x="7" y="11" width="3" height="6"/><rect x="13" y="7" width="3" height="10"/></svg></span>
      <span class="nl">분석</span>
    </button>
  </nav>"""

VERDICT_OLD = """      <div class="why" id="v-why">팀장님 신호를 읽고 있어요.</div>
    </div>"""
VERDICT_NEW = """      <div class="why" id="v-why">팀장님 신호를 읽고 있어요.</div>
      <button id="analyzeBtn" class="btn" style="margin-top:10px;background:var(--panel2);color:var(--tx);border:1px solid var(--line);padding:6px 12px;font-size:.78rem;border-radius:8px;cursor:pointer">&#128202; 상세 원인 분석</button>
    </div>"""

PANEL_OLD = """      <div class="ilist" id="ap-list"></div>
    </div>
  </div>"""
PANEL_NEW = """      <div class="ilist" id="ap-list"></div>
    </div>
  </div>

  <div class="panel" data-panel="analysis">
    <div class="pn" style="flex:1;min-height:0;overflow:auto">
      <div class="pt"><span>&#128202; 상세 원인 분석</span><span class="badge" id="anz-updated"></span></div>
      <div style="font-size:.78rem;color:var(--tx2);margin-bottom:10px;line-height:1.5">각 구간(1·5·10분)의 신호등이 알고리즘에서 어떻게 결정됐는지 단계별로 보여줍니다. 우선순위: ① 한숨 → ② 표정 대표값 → ③ 발화 부정 개수 → ④ 기본값(노랑). 굵게 표시된 단계에서 신호가 확정됩니다.</div>
      <div id="anz-cards"></div>
    </div>
  </div>"""

ANALYSIS_JS = """  /* ───────── 상세 원인 분석 탭 ───────── */
  function analyzeWindow(k,rd,th){
    var steps=[], fired=0;
    if(rd.sigh){steps.push({t:'① 한숨',d:'구간 내 한숨 1회 이상',res:'🔴 빨강 (최우선)',on:true});fired=1;}
    else steps.push({t:'① 한숨',d:'감지 없음',res:'다음 단계 ↓',on:false});
    if(!fired){
      if(rd.expr==='분노'){steps.push({t:'② 표정 대표값',d:'분노',res:'🔴 빨강',on:true});fired=2;}
      else if(rd.expr==='행복'){steps.push({t:'② 표정 대표값',d:'행복',res:'🟢 초록',on:true});fired=2;}
      else steps.push({t:'② 표정 대표값',d:(rd.expr||'일반')+' (보통)',res:'다음 단계 ↓',on:false});
    }else steps.push({t:'② 표정 대표값',d:(rd.expr||'-'),res:'평가 안 함',on:false});
    if(!fired){
      if(rd.neg_count>=th){steps.push({t:'③ 발화 부정',d:rd.neg_count+'개 ≥ 임계 '+th+'개',res:'🔴 빨강',on:true});fired=3;}
      else steps.push({t:'③ 발화 부정',d:rd.neg_count+'개 < 임계 '+th+'개',res:'다음 단계 ↓',on:false});
    }else steps.push({t:'③ 발화 부정',d:rd.neg_count+'개',res:'평가 안 함',on:false});
    if(!fired)steps.push({t:'④ 기본값',d:'위 조건 모두 불충족',res:'🟡 노랑',on:true});
    return steps;
  }
  function renderAnalysis(){
    var box=document.getElementById('anz-cards'); if(!box)return;
    fetch('/api/radar/signals').then(function(r){return r.json();}).then(function(d){
      if(d.error){box.innerHTML='<div class="empty">radar 미연결</div>';return;}
      var up=document.getElementById('anz-updated');
      if(up&&d.updated_at)up.textContent='업데이트 '+d.updated_at.substring(11,19);
      var th={'1m':1,'5m':2,'10m':3}, lbl={'1m':'최근 1분','5m':'최근 5분','10m':'최근 10분'};
      var sigKo={red:'🔴 빨강',yellow:'🟡 노랑',green:'🟢 초록'};
      var html='';
      ['1m','5m','10m'].forEach(function(k){
        var rd=d[k]; if(!rd)return;
        var dp=rd.data_points||{expression:0,sigh:0,speech:0};
        var notReady=(rd.ready===false)||(dp.expression===0);
        html+='<div style="border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px;background:var(--panel2)">';
        html+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><strong style="font-size:.92rem">'+lbl[k]+'</strong><span style="font-weight:800">'+(notReady?'⏳ 측정중':(sigKo[rd.signal]||rd.signal))+'</span></div>';
        if(notReady){
          html+='<div style="font-size:.78rem;color:var(--tx2)">구간이 아직 충분히 누적되지 않았습니다 (표정 '+dp.expression+'개).</div>';
        }else{
          analyzeWindow(k,rd,th[k]).forEach(function(s){
            var c=s.on?'var(--tx)':'var(--tx3)';
            html+='<div style="display:flex;gap:8px;padding:5px 0;border-top:1px solid var(--line2);font-size:.8rem"><span style="flex:0 0 120px;color:'+c+';font-weight:'+(s.on?'700':'400')+'">'+s.t+'</span><span style="flex:1;color:var(--tx2)">'+s.d+'</span><span style="flex:0 0 auto;font-weight:'+(s.on?'700':'400')+'">'+s.res+'</span></div>';
          });
          html+='<div style="font-size:.72rem;color:var(--tx3);margin-top:8px">샘플 — 표정 '+dp.expression+' · 한숨 '+dp.sigh+' · 발화 '+dp.speech+'</div>';
        }
        html+='</div>';
      });
      box.innerHTML=html||'<div class="empty">데이터 없음</div>';
    }).catch(function(){box.innerHTML='<div class="empty">신호 데이터를 불러올 수 없습니다.</div>';});
  }
  var _aBtn=document.getElementById('analyzeBtn');
  if(_aBtn)_aBtn.addEventListener('click',function(){showTab('analysis');renderAnalysis();});
  setInterval(function(){if(curTab==='analysis')renderAnalysis();},3000);
"""

# ── 패치 목록: (설명, old, new, marker) ──
PATCHES = [
  ("측정중 판정에 ready 반영",
   "if(!rd||rd.data_points.expression===0){lastStatus={signal:'measuring',window:w};applyStatusUI(lastStatus);return;}",
   "if(!rd||rd.ready===false||rd.data_points.expression===0){lastStatus={signal:'measuring',window:w};applyStatusUI(lastStatus);return;}",
   "rd.ready===false||rd.data_points.expression===0"),
  ("측정중 v-why 제거",
   "document.getElementById('v-why').textContent=(lbl?lbl+' ':'')+'누적 데이터를 모으는 중이에요. 잠시만 기다려 주세요.';",
   "document.getElementById('v-why').textContent='';",
   "document.getElementById('v-why').textContent='';"),
  ("측정중 배지 제거",
   "var cb=document.getElementById('clr-badge'); if(cb){cb.textContent='⏳ 측정중';cb.className='clr-badge clr-wait';}",
   "var cb=document.getElementById('clr-badge'); if(cb){cb.textContent='';cb.className='clr-badge';}",
   "if(cb){cb.textContent='';cb.className='clr-badge';}"),
  ("측정중 mood-sub 제거",
   "var rms=document.getElementById('r-mood-sub'); if(rms) rms.textContent='데이터 수집 중';",
   "var rms=document.getElementById('r-mood-sub'); if(rms) rms.textContent='';",
   "r-mood-sub'); if(rms) rms.textContent='';"),
  ("측정중 time-sub 제거",
   "var rts=document.getElementById('r-time-sub'); if(rts) rts.textContent='측정중';",
   "var rts=document.getElementById('r-time-sub'); if(rts) rts.textContent='';",
   "r-time-sub'); if(rts) rts.textContent='';"),
  ("캘리브레이션 모달 삽입(레벨 막대 포함)",
   "<script>\n(function(){",
   CALIB_MODAL + "<script>\n(function(){",
   'id="calib-level-bar"'),
  ("음성 캘리브레이션 핸들러 교체", OLD_VCALIB, NEW_VCALIB, "var _calibTimer=null;"),
  ("분석 탭 사이드바 추가", NAV_OLD, NAV_NEW, 'data-tab="analysis"'),
  ("상세 원인 분석 버튼 추가", VERDICT_OLD, VERDICT_NEW, 'id="analyzeBtn"'),
  ("분석 패널 추가", PANEL_OLD, PANEL_NEW, 'data-panel="analysis"'),
  ("TABS 배열에 analysis 추가",
   "var TABS=['judge','cal','ap'], curTab='judge';",
   "var TABS=['judge','cal','ap','analysis'], curTab='judge';",
   "['judge','cal','ap','analysis']"),
  ("showTab 에 analysis 분기",
   "if(name==='judge'){requestAnimationFrame(function(){drawGraph();});}\n  }",
   "if(name==='judge'){requestAnimationFrame(function(){drawGraph();});}\n    if(name==='analysis'){renderAnalysis();}\n  }",
   "if(name==='analysis'){renderAnalysis();}"),
  ("분석 JS 주입",
   "  loadCalendar(); loadApprovals();\n})();",
   "  loadCalendar(); loadApprovals();\n" + ANALYSIS_JS + "})();",
   "function renderAnalysis("),
]


def main():
    if not os.path.exists(PATH):
        sys.exit(f"❌ 파일 없음: {PATH}")
    with open(PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # v1 흔적 감지 (모달은 있는데 레벨 막대가 없으면 이전 패치본) → 원본 복원 요청
    if 'id="calib-mo"' in html and 'id="calib-level-bar"' not in html:
        sys.exit("⚠ 이전 패치본(v1)으로 보입니다. 원본에서 실행하세요:\n"
                 f"   cp {PATH}.bak {PATH}   # 후 다시 실행")

    shutil.copyfile(PATH, PATH + ".bak")
    print(f"백업: {PATH}.bak\n")

    applied = skipped = 0
    for desc, old, new, marker in PATCHES:
        if marker in html:
            print(f"  ⏭  이미 반영됨 — {desc}"); skipped += 1; continue
        cnt = html.count(old)
        if cnt == 0:
            print(f"  ⚠  앵커 못 찾음 — {desc}"); skipped += 1; continue
        if cnt > 1:
            print(f"  ⚠  앵커 {cnt}곳 — {desc} (모두 치환)")
        html = html.replace(old, new)
        print(f"  ✅ 반영 — {desc}"); applied += 1

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n완료: {applied}건 반영, {skipped}건 건너뜀 → {PATH}")
    if applied == 0:
        print("아무것도 반영되지 않았습니다. 원본인지 확인하세요.")


if __name__ == "__main__":
    main()