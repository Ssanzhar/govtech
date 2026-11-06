/* An AudioContext that says when it stops. A phone suspends it on an interruption (another app, an
   incoming call), and nothing resumes it by itself: the page kept saying "listening" over silence,
   and a low meter at the end read as reassurance. */

// Started after a long download the click may no longer count as a user gesture, and `resume()` can
// stay pending for as long as the interruption lasts, so it is never awaited unbounded.
const RESUME_WAIT_MS = 1000;
const STOPPED_STATES = new Set(["suspended", "interrupted"]);

/** A running context; `onInterrupted(true | false)` follows whether it is receiving audio. */
export async function startAudioContext(onInterrupted = () => {}, AudioContextImpl = globalThis.AudioContext) {
  const ctx = new AudioContextImpl();
  ctx.addEventListener("statechange", () => {
    if (ctx.state === "closed") return;
    const stopped = STOPPED_STATES.has(ctx.state);
    onInterrupted(stopped);
    if (stopped) ctx.resume().catch(() => {});
  });
  if (STOPPED_STATES.has(ctx.state)) {
    await Promise.race([ctx.resume().catch(() => {}), new Promise((resolve) => setTimeout(resolve, RESUME_WAIT_MS))]);
    if (STOPPED_STATES.has(ctx.state)) onInterrupted(true); // no statechange will say so
  }
  return ctx;
}
