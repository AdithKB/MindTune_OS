/* MindTune-OS — Dashboard JavaScript
 *
 * In plain English: this file drives everything you see moving on the dashboard.
 * It polls the Flask backend every 500 ms for the latest EEG state and updates
 * the charts, badges, and widgets in place — no page reloads needed.
 *
 * How data flows:
 *   main_loop.py  →  state.json  →  Flask /state endpoint  →  updateState()  →  DOM
 *   wins_log.json               →  Flask /memory           →  renderMemory()  →  DOM
 *
 * Key patterns used:
 *   Dirty-checking  — lastState{} caches the last value we wrote to the DOM.
 *                     We only touch the DOM when the value has actually changed.
 *                     This avoids 2 DOM writes per second × 30 widgets = 60 writes/s.
 *   Optimistic UI   — mode switches and feedback update the UI before the server
 *                     confirms, so the interface feels instant even over localhost.
 *   Sync lock       — isSwitchingMode pauses all state updates during a mode
 *                     transition to prevent a stale heartbeat from overwriting
 *                     the in-progress switch animation.
 */

// ── Module-level globals ──────────────────────────────────────────
let lastUpdateTime          = null;   // Unix ms of the last successful /state poll
let isStartingUp            = true;   // True until first /state response arrives
let startupStartTime        = Date.now();
let startupDismissScheduled = false;
let isSwitchingMode         = false;  // True while a Focus/Calm mode switch is in flight
let isSwitchingModeStarted  = 0;      // ms timestamp when switch began (for 10 s timeout)
let feedbackCooldown        = false;  // Prevents spamming the 👎 skip button
let lastBlinkTs             = 0;
let blinkResetTimer         = null;
let timelineChart           = null;   // Chart.js instance — created once in initChart()
window._lastFocusModeActive = false;  // Mirrors backend focus_mode_active for optimistic UI
let heartbeatTick           = 0;      // Increments every 500 ms; used to stagger poll rates
let lastState = {};                   // Dirty-check cache: stores last value written to each DOM element

// escHtml — XSS prevention for innerHTML assignments.
// Spotify track names, artist names, and Last.fm tags all come from external
// services and could contain HTML characters. Escaping before inserting into
// innerHTML ensures they're displayed as plain text, never executed as HTML.
function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// toggleFocusMode — Optimistic UI pattern.
// We update the button and body class IMMEDIATELY (before the server responds)
// so the transition feels instant. The sync lock (isSwitchingMode) then holds
// all updateState() calls until the backend confirms the switch is complete.
function toggleFocusMode() {
  const isFocusActive = window._lastFocusModeActive || false;
  const btn        = document.getElementById('mode-toggle-btn');
  const chip       = document.getElementById('mode-chip');
  const icon       = document.getElementById('mode-toggle-icon');
  const text       = document.getElementById('mode-toggle-text');
  const loader     = document.getElementById('mode-loader');
  const loaderText = document.getElementById('loader-text');

  isSwitchingMode = true;
  isSwitchingModeStarted = Date.now();

  if (loader) {
    loaderText.textContent = isFocusActive ? 'Switching to Calm Mode...' : 'Switching to Focus Mode...';
    loader.classList.add('visible');
  }

  if (isFocusActive) {
    document.body.classList.remove('focus-mode');
    if (icon) icon.textContent = '🧘';
    if (text) text.textContent = 'Switching to Calm Mode...';
    if (chip) chip.textContent  = '⇄ CALM MODE';
    window._lastFocusModeActive = false;
  } else {
    document.body.classList.add('focus-mode');
    if (icon) icon.textContent = '🎯';
    if (text) text.textContent = 'Switching to Focus Mode...';
    if (chip) chip.textContent  = '⇄ FOCUS MODE';
    window._lastFocusModeActive = true;
  }

  if (btn) {
    btn.disabled = true;
    const btnText = btn.querySelector('.mode-toggle-text');
    if (btnText) btnText.textContent = 'Switching...';
  }

  fetch(isFocusActive ? '/focus/off' : '/focus/on', { method: 'POST' })
    .then(() => { /* unlock happens on next successful pollState */ })
    .catch(e => {
      console.error('focus toggle failed:', e);
      isSwitchingMode = false;
      if (loader) loader.classList.remove('visible');
      if (btn) btn.disabled = false;
      if (text) text.textContent = 'Offline';
    });
}

// ── Feedback buttons ─────────────────────────────────────────────
function simulateDoubleBlink() {
  fetch('/blink/double', { method: 'POST' })
    .catch(e => console.warn('Double-blink simulation failed', e));
}

function forceStress() {
  fetch('/feedback/force_stress', { method: 'POST' })
    .then(() => {
      const btn = document.getElementById('force-stress-btn');
      if (btn) {
        btn.classList.add('active');
        setTimeout(() => btn.classList.remove('active'), 1000);
      }
    })
    .catch(e => console.error('Force stress failed:', e));
}

function recalibrateSensor() {
  const btn = document.getElementById('recalibrate-btn');
  if (btn) btn.disabled = true;
  
  fetch('/hardware/recalibrate', { method: 'POST' })
    .then(() => {
      if (btn) {
        const originalText = btn.textContent;
        btn.textContent = '🔄 Calibrating...';
        setTimeout(() => {
          btn.disabled = false;
          btn.textContent = originalText;
        }, 8000);
      }
    })
    .catch(e => {
      console.error('Recalibration failed:', e);
      if (btn) btn.disabled = false;
    });
}

function sendFeedback(action) {
  if (feedbackCooldown) return;

  const btnUp   = document.getElementById('btn-up');
  const btnDown = document.getElementById('btn-down');
  const confirm = document.getElementById('fb-confirm');

  btnUp.classList.remove('active');
  btnDown.classList.remove('active');
  const activeBtn = (action === 'up' ? btnUp : btnDown);
  activeBtn.classList.add('active');

  activeBtn.style.transform = 'scale(0.95)';
  setTimeout(() => { activeBtn.style.transform = ''; }, 100);

  if (action === 'down') {
    feedbackCooldown = true;
    setTimeout(() => { feedbackCooldown = false; }, 1500);
  }

  confirm.textContent = action === 'up' ? '👍 Saved win...' : '↓ Skipping...';
  confirm.style.opacity = '1';

  fetch(`/feedback/${action}`, { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      setTimeout(() => { confirm.style.opacity = '0'; }, 2000);
    })
    .catch(() => {
      confirm.textContent = 'Offline';
      confirm.style.opacity = '1';
    });
}

// ── Startup loader dismiss helper ────────────────────────────────
// STARTUP_MIN_MS — why we need a minimum display time:
// On a fast local server, /state resolves in ~100 ms and the loader would
// disappear before the user even saw it, making the app look like it skipped
// loading entirely. 1500 ms ensures the animation is always visible.
// startupDismissScheduled prevents stacked timeouts if pollState() fires
// multiple times while the loader is still counting down.
const STARTUP_MIN_MS = 1500;
function dismissStartupLoader() {
  if (!isStartingUp || startupDismissScheduled) return;
  const elapsed   = Date.now() - startupStartTime;
  const remaining = STARTUP_MIN_MS - elapsed;
  if (remaining > 0) {
    startupDismissScheduled = true;
    setTimeout(() => { startupDismissScheduled = false; dismissStartupLoader(); }, remaining);
    return;
  }
  isStartingUp = false;
  const startLoader = document.getElementById('startup-loader');
  if (startLoader) {
    startLoader.classList.add('fade-out'); // Optional CSS hook
    startLoader.classList.remove('visible');
    setTimeout(() => { startLoader.style.display = 'none'; }, 600);
  }
}

// ── State update ─────────────────────────────────────────────────
function updateState(data) {
  if (!data) return;

  dismissStartupLoader();

  // Clear skeleton state on first real data
  if (isStartingUp === false && lastState.firstLoadCleared !== true) {
    document.querySelectorAll('.skeleton').forEach(el => el.classList.remove('skeleton'));
    lastState.firstLoadCleared = true;
  }

  lastUpdateTime = Date.now();

  try {
    // ── Synchronization lock ────────────────────────────────────────
    // While a mode switch is in progress, the backend state briefly disagrees
    // with what the UI is showing (e.g. UI says "Focus" but backend hasn't
    // confirmed yet). We pause all DOM updates here until the backend catches up,
    // or until 10 seconds pass (safety timeout for network failures).
    if (isSwitchingMode) {
      const targetFocus = window._lastFocusModeActive;
      const actualFocus = !!data.focus_mode_active;
      const targetMode  = targetFocus ? 'focus' : 'calm';
      const actualMode  = data.current_mode || 'calm';
      const timedOut    = (Date.now() - isSwitchingModeStarted) > 10000;

      if ((targetFocus === actualFocus && targetMode === actualMode) || timedOut) {
        isSwitchingMode = false;
        const loader = document.getElementById('mode-loader');
        const btn    = document.getElementById('mode-toggle-btn');
        if (loader) loader.classList.remove('visible');
        if (btn) btn.disabled = false;
      } else {
        return;
      }
    }

    const inFocusContext = data.focus_mode_active || data.current_mode === 'focus';

    // ── Learning Badge ─────────────────────────────────────────────
    const phase   = inFocusContext ? (data.focus_ml_phase || 1) : (data.pref_phase || 1);
    const entries = inFocusContext ? (data.focus_ml_entries || 0) : (data.feedback_count || 0);
    const learningKey = `${phase}-${entries}-${inFocusContext}`;
    if (lastState.learning !== learningKey) {
      const learningBadge = document.getElementById('learning-badge');
      if (learningBadge) {
        if (phase === 2) {
          learningBadge.innerHTML = '🧠 <span class="badge-text">Personalized AI Model Active</span>';
          learningBadge.className = 'learning-badge phase-2';
        } else {
          const remaining = Math.max(0, 10 - entries);
          learningBadge.innerHTML = `⭐ <span class="badge-text">Training AI: ${remaining} more feedback${remaining === 1 ? '' : 's'} to activate ML</span>`;
          learningBadge.className = 'learning-badge phase-1';
        }
        learningBadge.style.display = 'block';
      }
      lastState.learning = learningKey;
    }

    // ── State chip ──────────────────────────────────────────────────
    const pred = data.prediction || 'other';
    const chipKey = `${pred}-${data.music_active}`;
    if (lastState.chipKey !== chipKey) {
      const chip = document.getElementById('state-chip');
      if (chip) {
        if (pred === 'stressed' && data.music_active) {
          chip.textContent = 'STRESSED · 🎵 INTERVENING';
          chip.className = 'state-chip stressed';
        } else {
          chip.textContent = pred.toUpperCase();
          chip.className = 'state-chip ' + (['stressed','calm','relaxed'].includes(pred) ? pred : 'other');
        }
      }
      lastState.chipKey = chipKey;
    }

    // ── EEG Dots (last 5 readings) ────────────────────────────────
    const dotsKey = (data.recent_predictions || []).join(',');
    if (lastState.dotsKey !== dotsKey) {
      const dotsRow = document.getElementById('dots-row');
      if (dotsRow) {
        while (dotsRow.firstChild) dotsRow.removeChild(dotsRow.firstChild);
        for (const p of (data.recent_predictions || [])) {
          const dot = document.createElement('div');
          const validPreds = ['calm', 'relaxed', 'stressed'];
          dot.className = 'dot ' + (validPreds.includes(p) ? 'dot-' + p : 'dot-unknown');
          dot.title = validPreds.includes(p) ? p : 'unknown';
          dotsRow.appendChild(dot);
        }
      }
      lastState.dotsKey = dotsKey;
    }

    // ── Confidence badge ──────────────────────────────────────────
    if (lastState.confidence !== data.confidence) {
      const confBadge = document.getElementById('conf-badge');
      const confVal   = document.getElementById('conf-val');
      if (confBadge && confVal && data.confidence !== undefined) {
        confVal.textContent = data.confidence.toFixed(2);
        confBadge.style.display = 'block';
        confVal.style.color = data.confidence > 0.8 ? 'var(--green)' : (data.confidence < 0.4 ? 'var(--red)' : 'var(--text)');
      }
      lastState.confidence = data.confidence;
    }

    // ── Mode toggle sync & Theme ──────────────────────────────────
    const modeKey = `${data.focus_mode_active}-${data.current_mode}`;
    if (lastState.modeKey !== modeKey) {
      const modeChip       = document.getElementById('mode-chip');
      const modeToggleBtn  = document.getElementById('mode-toggle-btn');
      const modeToggleIcon = document.getElementById('mode-toggle-icon');
      const modeToggleText = document.getElementById('mode-toggle-text');
      const banner         = document.getElementById('focus-banner');
      const agentSub       = document.getElementById('agent-panel-sub');
      const npSub          = document.getElementById('nowplaying-panel-sub');

      window._lastFocusModeActive = !!data.focus_mode_active;

      if (inFocusContext) {
        document.body.classList.add('focus-mode');
        if (banner) banner.classList.add('visible');
        if (modeChip) { modeChip.textContent = '⇄ FOCUS MODE'; modeChip.className = 'mode-chip focus'; }
        if (modeToggleIcon) modeToggleIcon.textContent = '🎯';
        if (modeToggleText) modeToggleText.textContent = 'FOCUS MODE — click to switch back to Calm';
        if (modeToggleBtn) modeToggleBtn.className = 'mode-toggle-btn focus-active';
        if (agentSub) agentSub.textContent = 'Why the AI chose this focus music — learns from past focus sessions';
        if (npSub)    npSub.textContent    = 'Instrumental music selected to protect your attention span';
      } else {
        document.body.classList.remove('focus-mode');
        if (banner) banner.classList.remove('visible');
        if (modeChip) { modeChip.textContent = '⇄ CALM MODE'; modeChip.className = 'mode-chip calm'; }
        if (modeToggleIcon) modeToggleIcon.textContent = '🧘';
        if (modeToggleText) modeToggleText.textContent = 'CALM MODE — click to switch to Focus Mode';
        if (modeToggleBtn) modeToggleBtn.className = 'mode-toggle-btn';
        if (agentSub) agentSub.textContent = 'Why the AI chose this music — learns from past wins & failures';
        if (npSub)    npSub.textContent    = 'Music selected by the AI to lower your stress level';
      }
      lastState.modeKey = modeKey;
    }

    // ── System Goal strip ─────────────────────────────────────────
    const goalKey = `${data.pending_win}-${data.music_active}-${data.stress_count}-${inFocusContext}`;
    if (lastState.goalKey !== goalKey) {
      const goalStrip = document.getElementById('goal-strip');
      const goalIcon  = document.getElementById('goal-icon');
      const goalText  = document.getElementById('goal-text');
      if (goalStrip && goalIcon && goalText) {
        if (data.pending_win) {
          goalStrip.className = 'goal-strip pending-win';
          goalIcon.textContent = '🏆';
          goalText.textContent = `Goal Reached: ${inFocusContext ? 'Focused' : 'Calm'} state detected. Saving win...`;
        } else if (inFocusContext) {
          if (data.music_active) {
            goalStrip.className = 'goal-strip focus-active';
            goalIcon.textContent = '🎯';
            goalText.textContent = 'Transition: Unfocused ➔ Focused';
          } else {
            goalStrip.className = 'goal-strip';
            goalIcon.textContent = '🔍';
            goalText.textContent = 'Focus Mode: Monitoring for attention drift...';
          }
        } else if (data.music_active) {
          goalStrip.className = 'goal-strip intervening';
          goalIcon.textContent = '🎵';
          goalText.textContent = 'Transition: Stressed ➔ Calm';
        } else if (data.stress_count >= 3) {
          goalStrip.className = 'goal-strip intervening';
          goalIcon.textContent = '⚡';
          goalText.textContent = 'Stress Peak Detected — selecting music...';
        } else {
          goalStrip.className = 'goal-strip';
          goalIcon.textContent = '🔍';
          goalText.textContent = 'Calm Mode: Monitoring for stress signals...';
        }
      }
      lastState.goalKey = goalKey;
    }

    // ── Pending Win banner ────────────────────────────────────────
    if (lastState.pending_win !== data.pending_win) {
      const pendingBanner = document.getElementById('pending-win-banner');
      if (pendingBanner) {
        if (data.pending_win) pendingBanner.classList.add('visible');
        else pendingBanner.classList.remove('visible');
      }
      lastState.pending_win = data.pending_win;
    }

    // ── Blink indicator ───────────────────────────────────────────
    if (data.last_blink_ts && data.last_blink_ts > lastBlinkTs) {
      lastBlinkTs = data.last_blink_ts;
      const blinkEl    = document.getElementById('blink-indicator');
      const blinkTxtEl = document.getElementById('blink-text');
      if (blinkEl) {
        blinkEl.classList.add('fired');
        if (blinkTxtEl) blinkTxtEl.textContent = '✓ Skip confirmed';
        clearTimeout(blinkResetTimer);
        blinkResetTimer = setTimeout(() => {
          blinkEl.classList.remove('fired');
          if (blinkTxtEl) blinkTxtEl.textContent = '👁 Blink remote';
        }, 1800);
      }
    }

    // ── Stress gauge ring (SVG) ───────────────────────────────────
    // gauge-fill stroke-dashoffset: 351.86 × (1 - stress/5)
    // Full circle = 0 offset; empty circle = 351.86 offset.
    const gaugeKey = `${data.stress_count}-${data.prediction}-${data.focus_mode_active}-${data.pending_win}`;
    if (lastState.gaugeKey !== gaugeKey) {
      const stressCount = Math.min(data.stress_count || 0, 5);
      const CIRC = 351.86;

      const gaugeEl = document.getElementById('gauge-fill');
      if (gaugeEl) {
        gaugeEl.style.strokeDashoffset = CIRC * (1 - stressCount / 5);
        if (data.focus_mode_active) {
          gaugeEl.style.stroke = 'var(--focus-col)';
        } else if (stressCount >= 3) {
          gaugeEl.style.stroke = 'var(--stress-col)';
        } else {
          gaugeEl.style.stroke = 'var(--calm-col)';
        }
      }

      // Update just the text node — the /5 <span class="gauge-denom"> stays in place
      const gaugeValEl = document.getElementById('gauge-val');
      if (gaugeValEl && gaugeValEl.firstChild && gaugeValEl.firstChild.nodeType === Node.TEXT_NODE) {
        gaugeValEl.firstChild.textContent = String(stressCount);
      }

      const explainEl = document.getElementById('state-explain');
      if (explainEl) {
        let msg;
        if (data.focus_mode_active)                       msg = 'Focus mode — protecting attention';
        else if (data.pending_win)                        msg = 'Stress dropping — saving win...';
        else if (data.music_active && stressCount >= 3)   msg = 'Music intervening — stress should drop';
        else if (stressCount >= 3)                        msg = 'High stress — AI is selecting music...';
        else if (data.prediction === 'stressed')          msg = 'Elevated stress — monitoring closely';
        else if (data.prediction === 'relaxed')           msg = 'Deeply relaxed — optimal state';
        else if (data.prediction === 'calm')              msg = 'Calm — brainwaves look good';
        else if (data.prediction === 'focused')           msg = 'Focused — cognitive load elevated';
        else if (data.prediction === 'waiting')           msg = 'Waiting for EEG signal...';
        else                                              msg = 'Monitoring brainwaves...';
        explainEl.textContent = msg;
      }

      lastState.gaugeKey = gaugeKey;
    }

    // ── Agent reasoning ───────────────────────────────────────────
    if (lastState.reasoning !== data.agent_reasoning) {
      const reasoningEl = document.getElementById('reasoning');
      if (reasoningEl) reasoningEl.textContent = data.agent_reasoning || '—';
      lastState.reasoning = data.agent_reasoning;
    }

    // ── Wins Count & Session ──────────────────────────────────────
    if (lastState.wins_count !== data.wins_count) {
      const winsCountEl = document.getElementById('wins-count');
      if (winsCountEl) winsCountEl.textContent = data.wins_count || 0;
      lastState.wins_count = data.wins_count;
    }

    // ── Signal Integrity Warning ──────────────────────────────────
    // signal_saturated is only set true by Arduino hardware source, never by CSV replay.
    // When saturated: show banner, hide bands/neuro-metrics (values are meaningless noise).
    // When not saturated (CSV or healthy hardware): show everything normally.
    if (lastState.signal_saturated !== data.signal_saturated) {
      const banner    = document.getElementById('saturation-banner');
      const strip     = document.getElementById('band-strip');
      const neuroEl   = document.getElementById('neuro-metrics');
      const confBadge = document.getElementById('conf-badge');
      const sat = !!data.signal_saturated;
      if (banner)    sat ? banner.classList.add('visible')   : banner.classList.remove('visible');
      if (strip)     strip.style.display     = sat ? 'none' : '';
      if (neuroEl)   neuroEl.style.display   = sat ? 'none' : '';
      if (confBadge) confBadge.style.display = sat ? 'none' : 'block';
      lastState.signal_saturated = data.signal_saturated;
    }

    // ── Debug Mode ────────────────────────────────────────────────
    const urlParams = new URLSearchParams(window.location.search);
    const isDebug   = urlParams.has('debug');
    if (lastState.isDebug !== isDebug) {
      const debugBtn = document.getElementById('force-stress-btn');
      if (debugBtn) {
        debugBtn.style.display = isDebug ? 'flex' : 'none';
      }
      lastState.isDebug = isDebug;
    }

    if (lastState.session_number !== data.session_number) {
      const sessionBadge = document.getElementById('session-badge');
      if (sessionBadge && data.session_number) {
        sessionBadge.textContent = `Session ${data.session_number}`;
        sessionBadge.style.display = '';
      }
      lastState.session_number = data.session_number;
    }

    // ── Now Playing ───────────────────────────────────────────────
    const np = data.now_playing || {};
    const npKey = `${np.track}-${np.artist}-${data.music_active}-${inFocusContext}`;
    if (lastState.npKey !== npKey) {
      const hasTrack = np.track && np.track !== 'Nothing playing';
      const trackNameEl  = document.getElementById('track-name');
      const artistNameEl = document.getElementById('artist-name');
      const spotifyLink  = document.getElementById('spotify-link');
      
      if (trackNameEl)  trackNameEl.textContent  = hasTrack ? np.track  : '—';
      if (artistNameEl) artistNameEl.textContent = hasTrack ? (np.artist || '') : '';
      
      if (spotifyLink) {
        if (hasTrack && np.uri && np.uri.startsWith('spotify:track:')) {
          spotifyLink.href = `https://open.spotify.com/track/${escHtml(np.uri.replace('spotify:track:', ''))}`;
          spotifyLink.style.display = '';
        } else {
          spotifyLink.style.display = 'none';
        }
      }

      const bars = ['mb1','mb2','mb3'].map(id => document.getElementById(id));
      const statusEl = document.getElementById('music-status');
      const btnUp    = document.getElementById('btn-up');
      const btnDown  = document.getElementById('btn-down');

      if (statusEl) {
        if (data.music_active) {
          bars.forEach(b => b && b.classList.add('on'));
          statusEl.textContent = inFocusContext ? 'Focusing' : 'Calming';
          statusEl.className   = 'music-status';
          if (inFocusContext) statusEl.classList.add('focus-active');
          if (!feedbackCooldown && btnUp && btnDown) { btnUp.disabled = false; btnDown.disabled = false; }
        } else if (hasTrack) {
          bars.forEach(b => b && b.classList.add('on'));
          statusEl.textContent = inFocusContext ? 'Monitoring Focus' : 'Monitoring Calm';
          statusEl.className   = 'music-status monitoring';
          if (inFocusContext) statusEl.classList.add('focus-active');
          if (btnUp && btnDown) {
            btnUp.disabled = true; btnDown.disabled = true;
            btnUp.classList.remove('active'); btnDown.classList.remove('active');
          }
        } else {
          bars.forEach(b => b && b.classList.remove('on'));
          statusEl.innerHTML = 'Not playing · <span style="color:var(--text3); font-weight:normal;">Play music on Spotify</span>';
          statusEl.className = 'music-status';
          if (btnUp && btnDown) { btnUp.disabled = true; btnDown.disabled = true; }
        }
      }
      lastState.npKey = npKey;
    }

    if (lastState.status_message !== data.status_message) {
      const statusMsgEl = document.getElementById('status-msg');
      if (statusMsgEl) statusMsgEl.textContent = data.status_message || '';
      lastState.status_message = data.status_message;
    }

    // ── Tried tags ────────────────────────────────────────────────
    const triedKey = (data.interventions_tried || []).join(',');
    if (lastState.triedKey !== triedKey) {
      const tagsEl = document.getElementById('tried-tags');
      if (tagsEl) {
        const tried = [...new Set((data.interventions_tried || []).filter(q => q))];
        if (tried.length > 0) {
          tagsEl.innerHTML = [...tried].reverse().slice(0, 8)
            .map(q => `<span class="tried-tag">${escHtml(q)}</span>`).join('');
        } else {
          tagsEl.innerHTML = '<span style="font-size:0.64rem;color:var(--text3)">None yet</span>';
        }
      }
      lastState.triedKey = triedKey;
    }

    // ── Debug message ─────────────────────────────────────────────
    if (lastState.debug_msg !== data.debug_msg) {
      const debugEl = document.getElementById('debug-msg');
      if (debugEl && data.debug_msg) {
        debugEl.textContent = data.debug_msg;
        debugEl.style.display = 'block';
      }
      lastState.debug_msg = data.debug_msg;
    }

    // ── Cognitive Load Index (θ/β) + Alpha Dominance Index ───────────
    const neuroKey = `${data.cognitive_load}-${data.alpha_dominance}`;
    if (lastState.neuroKey !== neuroKey) {
      // Focus banner: show CLI while in focus mode
      const bannerRatio = document.getElementById('focus-banner-ratio');
      if (bannerRatio) bannerRatio.textContent = data.cognitive_load != null ? `· CLI: ${data.cognitive_load}` : '';

      // Neuro-metrics row in brain card: show both indices
      const neuroEl = document.getElementById('neuro-metrics');
      if (neuroEl) {
        if (data.cognitive_load != null || data.alpha_dominance != null) {
          neuroEl.innerHTML =
            `<span title="Cognitive Load Index (DASM Proxy) — θ/β ratio">CLI <strong>${data.cognitive_load ?? '—'}</strong></span>` +
            `<span title="Alpha Dominance Index — α/(β+γ), single-channel approach motivation proxy (Davidson, 1988)">ADI <strong>${data.alpha_dominance ?? '—'}</strong></span>`;
          neuroEl.style.display = 'flex';
        } else {
          neuroEl.style.display = 'none';
        }
      }
      lastState.neuroKey = neuroKey;
    }

    // ── EEG weights (brainwave → music influence bars) ────────────
    const weightsKey = JSON.stringify(data.eeg_weights);
    if (lastState.weightsKey !== weightsKey) {
      const eegBars    = document.getElementById('eeg-bars');
      const eegSection = document.getElementById('eeg-influence-section');
      const BAND_LABELS = { Delta:'δ Delta', Theta:'θ Theta', Alpha:'α Alpha', Beta:'β Beta', Gamma:'γ Gamma', 'α/β ratio':'α/β Ratio', 'θ/β ratio':'θ/β Ratio' };
      if (eegBars && eegSection && data.eeg_weights && data.eeg_weights.length > 0) {
        eegBars.innerHTML = data.eeg_weights.map(w => {
          const isPos = w.weight >= 0;
          const pct   = Math.round(w.strength * 100);
          return `<div class="eeg-bar-row">
            <span class="eeg-bar-name">${escHtml(BAND_LABELS[w.band] || w.band)}</span>
            <div class="eeg-bar-track"><div class="eeg-bar-fill ${isPos ? 'pos' : 'neg'}" style="width:${pct}%"></div></div>
            <span class="eeg-bar-val" style="color:${isPos ? 'var(--green)' : 'var(--red)'}">${isPos ? '+' : ''}${w.weight.toFixed(2)}</span>
          </div>`;
        }).join('');
        eegSection.style.display = '';
      } else if (eegSection) {
        eegSection.style.display = 'none';
      }
      lastState.weightsKey = weightsKey;
    }

    // ── ML top/worst tags ─────────────────────────────────────────
    const mlKey = JSON.stringify([data.ml_top_tags, data.ml_worst_tags]);
    if (lastState.mlKey !== mlKey) {
      const mlSec = document.getElementById('ml-tag-section');
      if (mlSec) {
        const mlTopEl   = document.getElementById('ml-top-tags');
        const mlWorstEl = document.getElementById('ml-worst-tags');
        let mlVisible   = false;
        if (mlTopEl && data.ml_top_tags && data.ml_top_tags.length > 0) {
          mlTopEl.innerHTML = data.ml_top_tags.map(t => `<span class="pref-tag good">${escHtml(t.tag)} <small>+${t.weight.toFixed(2)}</small></span>`).join('');
          document.getElementById('ml-top-row').style.display = 'flex';
          mlVisible = true;
        }
        if (mlWorstEl && data.ml_worst_tags && data.ml_worst_tags.length > 0) {
          mlWorstEl.innerHTML = data.ml_worst_tags.map(t => `<span class="pref-tag bad">${escHtml(t.tag)} <small>${t.weight.toFixed(2)}</small></span>`).join('');
          document.getElementById('ml-worst-row').style.display = 'flex';
          mlVisible = true;
        }
        mlSec.style.display = mlVisible ? '' : 'none';
      }
      lastState.mlKey = mlKey;
    }

  } catch (e) {
    console.error("Dashboard Update Error:", e);
  }
}

// ── Memory Log ───────────────────────────────────────────────────
function renderMemory(entries) {
  const container = document.getElementById('memory-log');
  if (!entries || entries.length === 0) {
    container.innerHTML = '<div class="memory-empty">No entries yet — first session</div>';
    return;
  }
  container.innerHTML = [...entries].reverse().slice(0, 10).map(e => {
    const isWin  = e.status === 'win';
    const badge  = isWin
      ? '<span class="badge badge-win">WIN</span>'
      : '<span class="badge badge-failed">FAIL</span>';
    const stats  = isWin
      ? `Stress ${escHtml(e.stress_before)}/5 → ${escHtml(e.stress_after)}/5 · resolved in ${escHtml(e.response_seconds)}s`
      : `Stress stayed at ${escHtml(e.stress_after)}/5 after ${escHtml(e.seconds_played)}s`;
    const ts     = e.timestamp ? e.timestamp.replace('T',' ').split('.')[0] : '';
    let efficacyHtml = '';
    if (isWin && e.response_seconds != null) {
      const secs = Number(e.response_seconds);
      const pct  = Math.max(10, Math.round(100 - (secs / 60) * 90));
      const spd  = secs <= 15 ? 'fast' : secs <= 30 ? 'moderate' : 'slow';
      efficacyHtml = `
        <div class="efficacy-wrap">
          <div class="efficacy-track">
            <div class="efficacy-fill" style="width:${pct}%"></div>
          </div>
          <span class="efficacy-label">${spd} · ${secs}s</span>
        </div>`;
    }
    return `
      <div class="memory-card ${escHtml(e.status)}">
        <div class="card-header">
          ${badge}
          <span class="card-track">${escHtml(e.track)}</span>
        </div>
        <div class="card-artist">${escHtml(e.artist)}</div>
        <div class="card-reason">${escHtml(e.reason)}</div>
        <div class="card-stats">${stats}</div>
        ${efficacyHtml}
        <div class="card-ts">${escHtml(ts)}</div>
      </div>`;
  }).join('');
}

// ── EEG Bands ────────────────────────────────────────────────────
const BAND_META = [
  { key: 'Delta', label: 'δ Delta', human: 'deep sleep' },
  { key: 'Theta', label: 'θ Theta', human: 'unfocused' },
  { key: 'Alpha', label: 'α Alpha', human: 'calm' },
  { key: 'Beta',  label: 'β Beta',  human: 'stressed' },
  { key: 'Gamma', label: 'γ Gamma', human: 'intense' },
];

// bandColor — interpolates between green (calm, score=0) and red (stressed, score=1).
// The start color rgb(16,185,129) is --green; the end color rgb(244,63,94) is --red.
// Each channel is lerped linearly: channel = start + (end - start) × score.
function bandColor(score) {
  const r = Math.round(16  + (244 - 16)  * score);
  const g = Math.round(185 + (63  - 185) * score);
  const b = Math.round(129 + (94  - 129) * score);
  return `rgb(${r},${g},${b})`;
}

function updateBands(bandScores) {
  try {
    const strip = document.getElementById('band-strip');
    if (!strip) return;
    if (!bandScores || Object.keys(bandScores).length === 0) {
      strip.style.display = 'none';
      return;
    }
    strip.style.display = 'flex';
    strip.innerHTML = BAND_META.map(m => {
      const score = bandScores[m.key] ?? 0;
      const pct   = Math.round(score * 100);
      const color = bandColor(score);
      return `
        <div class="band-item" title="${escHtml(m.label)} — ${pct}% toward stressed">
          <div class="band-name">${escHtml(m.label)}</div>
          <div class="band-bar-wrap">
            <div class="band-bar" style="width:${pct}%;background:${color};"></div>
          </div>
          <div class="band-human" style="color:${score > 0.5 ? color : 'var(--text2)'}">${escHtml(m.human)}</div>
          <div class="band-pct">${pct}%</div>
        </div>`;
    }).join('');
  } catch (e) {
    console.error("updateBands error:", e);
  }
}

// ── Live Pipeline Strip ──────────────────────────────────────────
function updatePipeline(data) {
  const pred        = data.prediction || '—';
  const stress      = data.stress_count ?? 0;
  const musicOn     = !!data.music_active;
  const wins        = data.wins_count ?? 0;
  const fbTotal     = data.feedback_count ?? 0;
  const connected   = pred !== 'waiting' && pred !== '—';

  // Step 1 — EEG Reading (always active when connected)
  const eegEl  = document.getElementById('ps-eeg');
  const eegVal = document.getElementById('ps-eeg-val');
  if (eegEl && eegVal) {
    eegEl.classList.toggle('active', connected);
    eegEl.classList.remove('stressed');
    eegVal.textContent = connected ? `${stress}/5 stress` : '—';
  }

  // Step 2 — Classify (shows current brain state)
  const clsEl  = document.getElementById('ps-classify');
  const clsVal = document.getElementById('ps-classify-val');
  if (clsEl && clsVal) {
    const isStressed = pred === 'stressed';
    clsEl.classList.toggle('active', connected && !isStressed);
    clsEl.classList.toggle('stressed', connected && isStressed);
    clsVal.textContent = connected ? pred.toUpperCase() : '—';
  }

  // Step 3 — Select Music (active while music is playing)
  const musEl  = document.getElementById('ps-music');
  const musVal = document.getElementById('ps-music-val');
  if (musEl && musVal) {
    musEl.classList.toggle('active', musicOn);
    musEl.classList.toggle('stressed', false);
    if (musicOn && data.now_playing && data.now_playing.track && data.now_playing.track !== 'Nothing playing') {
      const t = data.now_playing.track;
      musVal.textContent = t.length > 18 ? t.slice(0, 16) + '…' : t;
    } else {
      musVal.textContent = musicOn ? 'playing…' : 'waiting';
    }
  }

  // Step 4 — Learn (active when we have wins or feedback)
  const lrnEl  = document.getElementById('ps-learn');
  const lrnVal = document.getElementById('ps-learn-val');
  if (lrnEl && lrnVal) {
    const hasData = wins > 0 || fbTotal > 0;
    lrnEl.classList.toggle('active', hasData);
    lrnEl.classList.remove('stressed');
    lrnVal.textContent = hasData ? `${wins}W · ${fbTotal}FB` : 'no data yet';
  }
}

// Audio profile removed — Pure EEG mode, audio features not used by classifier

// ── Chart.js timeline ────────────────────────────────────────────
function initChart() {
  try {
    const ctx = document.getElementById('timeline-chart');
    if (!ctx) return;
    timelineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels:   [],
        datasets: [{
          data: [], borderWidth: 2, pointRadius: 0, tension: 0.5, fill: true,
          segment: {
            borderColor:     ctx => ctx.p1.parsed.y >= 3 ? '#f43f5e' : '#10b981',
            backgroundColor: ctx => ctx.p1.parsed.y >= 3
              ? 'rgba(244,63,94,0.1)' : 'rgba(16,185,129,0.07)',
          },
          borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.07)',
        }]
      },
      options: {
        animation: { duration: 600, easing: 'easeInOutCubic' },
        responsive: true,
        maintainAspectRatio: false, // Chart will now expand to fill the available panel height
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            ticks: { color: '#64748b', maxTicksLimit: 8,
                     callback: (_, i, vals) => vals[i]?.label ?? '' },
            grid:  { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            min: 0, max: 5,
            ticks: { color: '#64748b', stepSize: 1 },
            grid:  { color: 'rgba(255,255,255,0.04)' },
            title: { display: true, text: 'Stress (0–5)', color: '#64748b', font: { size: 10, weight: 'bold' } },
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { 
            enabled: true,
            backgroundColor: 'rgba(11, 13, 20, 0.9)',
            titleFont: { size: 11, weight: 'bold' },
            padding: 10,
            cornerRadius: 8,
            callbacks: { label: ctx => `Stress: ${ctx.parsed.y}/5` } 
          }
        }
      }
    });
  } catch (e) {
    console.error("Chart.js initialization failed:", e);
  }
}

window.onload = () => {
  // Initialization order matters:
  // 1. Add skeleton shimmer to all panels immediately — they fill in once data arrives.
  // 2. Create the Chart.js canvas before the first pollState() tries to draw into it.
  // 3. Fire all polls immediately so every widget has real data on first render.
  // 4. Start the 500 ms heartbeat for ongoing updates.
  // 5. Safety fallback: dismiss the startup loader after 4 s even if /state never responds.
  document.querySelectorAll('.panel, #timeline-chart').forEach(el => el.classList.add('skeleton'));
  initChart();
  pollState(); pollMemory(); pollSessions(); pollFeedback();
  setInterval(heartbeat, 500);
  setTimeout(dismissStartupLoader, 4000);
};

function updateTimeline(data) {
  try {
    if (!timelineChart || !data.timeline || data.timeline.length === 0) return;
    const accentColor = (data.current_mode === 'focus') ? '#818cf8' : '#10b981';
    const accentFill  = (data.current_mode === 'focus')
      ? 'rgba(129,140,248,0.1)' : 'rgba(16,185,129,0.1)';
    const ds = timelineChart.data.datasets[0];
    ds.segment = {
      borderColor:     ctx => ctx.p1.parsed.y >= 3 ? '#f43f5e' : accentColor,
      backgroundColor: ctx => ctx.p1.parsed.y >= 3 ? 'rgba(244,63,94,0.15)' : accentFill,
    };
    ds.borderColor     = accentColor;
    ds.backgroundColor = accentFill;
    timelineChart.data.labels = data.timeline.map(p => p.t);
    ds.data                   = data.timeline.map(p => p.stress);
    timelineChart.update();
  } catch (e) {
    console.error("updateTimeline error:", e);
  }
}

function renderSessions(sessions) {
  const container = document.getElementById('sessions-list');
  const trendEl   = document.getElementById('trend-line');
  if (!sessions || sessions.length === 0) {
    container.innerHTML = '<div class="no-sessions">No session data yet</div>';
    trendEl.style.display = 'none';
    return;
  }
  container.innerHTML = sessions.map(s => {
    const rt = s.avg_response_seconds !== null ? `avg ${s.avg_response_seconds}s` : '—';
    return `
      <div class="session-row">
        <div class="session-label">S${s.session}</div>
        <div class="session-bar-wrap">
          <div class="session-bar" style="width:${s.win_rate}%"></div>
        </div>
        <div class="session-stats">
          <span class="win-stat">${s.wins}W</span>/${s.fails}F &middot; ${rt}
        </div>
      </div>`;
  }).join('');

  const withWins = sessions.filter(s => s.avg_response_seconds !== null);
  if (withWins.length >= 2) {
    const prev = withWins[withWins.length - 2].avg_response_seconds;
    const curr = withWins[withWins.length - 1].avg_response_seconds;
    const diff = Math.round(curr - prev);
    trendEl.style.display = '';
    if (diff < 0)
      trendEl.innerHTML = `<span class="improving">↓ ${Math.abs(diff)}s faster</span> than last session`;
    else if (diff > 0)
      trendEl.innerHTML = `<span class="worsening">↑ ${diff}s slower</span> than last session`;
    else
      trendEl.innerHTML = 'Same avg resolution time as last session';
  } else if (withWins.length === 1) {
    trendEl.style.display = '';
    trendEl.innerHTML = `Baseline: ${withWins[0].avg_response_seconds}s avg`;
  } else {
    trendEl.style.display = 'none';
  }
}

setInterval(() => {
  const dot  = document.getElementById('live-dot');
  const text = document.getElementById('live-text');
  if (!dot || !text) return;
  if (!lastUpdateTime) {
    dot.classList.remove('on');
    text.textContent = 'Waiting...';
    return;
  }
  const secs = Math.round((Date.now() - lastUpdateTime) / 1000);
  if (secs < 5) {
    dot.classList.add('on');
    text.textContent = 'Live';
    text.style.color = '';
  } else if (secs < 60) {
    dot.classList.remove('on');
    text.textContent = `${secs}s ago`;
    text.style.color = '#f59e0b';
  } else {
    dot.classList.remove('on');
    text.textContent = `${Math.round(secs/60)}m ago`;
    text.style.color = '#f43f5e';
  }
}, 1000);

function pollState() {
  fetch('/state')
    .then(r => r.json())
    .then(data => {
      if (!data) return;
      try { updateTimeline(data); } catch(e) { console.error("Timeline update failed", e); }
      try { updateBands(data.band_scores); } catch(e) { console.error("Bands update failed", e); }
      try { updateState(data); } catch(e) { console.error("State update failed", e); }
      try { updatePipeline(data); } catch(e) { console.error("Pipeline update failed", e); }
    })
    .catch(err => {
      console.warn("Poll State failed (offline?)", err);
      dismissStartupLoader();
    });
}

function pollMemory() { fetch('/memory').then(r => r.json()).then(renderMemory).catch(() => {}); }
function pollSessions() { fetch('/sessions').then(r => r.json()).then(renderSessions).catch(() => {}); }

function renderFeedback(d) {
  if (!d) return;
  document.getElementById('pref-total').textContent = d.total    || 0;
  document.getElementById('pref-pos').textContent   = d.positive || 0;
  document.getElementById('pref-neg').textContent   = d.negative || 0;

  const phaseEl = document.getElementById('pref-phase');
  const hintEl  = document.getElementById('pref-hint');
  if (d.phase === 2) {
    phaseEl.textContent = 'Phase 2 — ML model active';
    phaseEl.className   = 'pref-phase-badge p2';
    hintEl.textContent  = 'Logistic regression trained on your EEG + music tag data';
  } else {
    const remaining = Math.max(0, 10 - (d.total || 0));
    phaseEl.textContent = 'Phase 1 — Frequency scoring';
    phaseEl.className   = 'pref-phase-badge p1';
    hintEl.textContent  = remaining > 0 ? `Give ${remaining} more feedback entries to activate ML` : 'ML model will activate soon';
  }

  // Keep the learning badge in the Now Playing card in sync with the authoritative /feedback phase
  const learningBadge = document.getElementById('learning-badge');
  if (learningBadge) {
    if (d.phase === 2) {
      learningBadge.innerHTML   = '🧠 <span class="badge-text">Personalized AI Model Active</span>';
      learningBadge.className   = 'learning-badge phase-2';
      learningBadge.style.display = 'block';
    } else {
      const remaining = Math.max(0, 10 - (d.total || 0));
      learningBadge.innerHTML   = `⭐ <span class="badge-text">Training AI: ${remaining} more feedback${remaining === 1 ? '' : 's'} to activate ML</span>`;
      learningBadge.className   = 'learning-badge phase-1';
      learningBadge.style.display = 'block';
    }
  }

  const topTagsEl  = document.getElementById('top-tags');
  const badTagsEl  = document.getElementById('bad-tags');
  if (d.top_tags && d.top_tags.length > 0) {
    topTagsEl.innerHTML = d.top_tags.map(t => `<span class="pref-tag good">${escHtml(t.tag)}</span>`).join('');
    document.getElementById('top-tags-row').style.display = 'flex';
  }
  if (d.worst_tags && d.worst_tags.length > 0) {
    badTagsEl.innerHTML = d.worst_tags.map(t => `<span class="pref-tag bad">${escHtml(t.tag)}</span>`).join('');
    document.getElementById('bad-tags-row').style.display = 'flex';
  }
  const emptyEl = document.getElementById('brain-patterns-empty');
  if (emptyEl) {
    const hasContent = (d.top_tags && d.top_tags.length > 0) || (d.worst_tags && d.worst_tags.length > 0);
    emptyEl.style.display = hasContent ? 'none' : '';
  }
}

function pollFeedback() { fetch('/feedback').then(r => r.json()).then(renderFeedback).catch(() => {}); }

// heartbeat — single timer that drives all polling at different rates.
// Everything runs off one 500 ms interval to avoid timer drift and make
// the update cadence easy to reason about:
//   Every tick   (500 ms) — /state       : EEG prediction, music status, stress level
//   Every 10     (  5  s) — /memory      : last 20 wins/fails
//                           /feedback    : preference model stats + tags
//   Every 30     ( 15  s) — /sessions    : per-session win-rate history
//   Every 60     ( 30  s) — /profile     : averaged audio features of winning tracks
// Slower endpoints are polled less often because their data changes infrequently
// and they read from files on disk — no need to hit them 2× per second.
async function heartbeat() {
  heartbeatTick++;
  pollState();
  if (heartbeatTick % 10 === 0) { pollMemory(); pollFeedback(); }
  if (heartbeatTick % 30 === 0) { pollSessions(); }
}
