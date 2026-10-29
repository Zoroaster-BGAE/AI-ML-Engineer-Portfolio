import type { StreamEvent } from "./types";

const TOKEN_KEY = "assistant.token";
const REFRESH_KEY = "assistant.refresh";

/**
 * Cognito hands back two tokens with very different lifetimes: an id token good
 * for five minutes, which every API call carries, and a refresh token good for
 * thirty days, which exists only to get another id token.
 *
 * Five minutes is deliberate, and it is the whole revocation story. The API
 * verifies a signature offline, so it cannot know an account was disabled --
 * a token already signed stays valid until it expires. Refreshing is the step
 * that checks: it is a call to Cognito, and Cognito refuses it for a disabled
 * user. So the shorter the id token, the tighter the window, at the cost of a
 * refresh every few minutes.
 *
 * Which is why a 401 here means "refresh", not "log out".
 */
export const tokenStore = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } },
  set: (t: string) => { try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode: session-only */ } },
  getRefresh: () => { try { return localStorage.getItem(REFRESH_KEY); } catch { return null; } },
  setRefresh: (t: string) => { try { localStorage.setItem(REFRESH_KEY, t); } catch { /* ignore */ } },
  clear: () => {
    try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REFRESH_KEY); } catch { /* ignore */ }
  },
};

/**
 * Pool details, read once from the API. Not secret: both ids travel in every login.
 * `endpoint` is set only in local development, where a Cognito emulator stands in for AWS.
 */
export interface AuthConfig {
  user_pool_id: string; client_id: string; region: string; configured: boolean; endpoint?: string | null;
}

let configPromise: Promise<AuthConfig> | null = null;
export function authConfig(): Promise<AuthConfig> {
  // Cached as the promise, not the value, so concurrent callers during startup
  // share one request instead of racing several.
  configPromise ??= fetch("/api/auth/config").then((r) => {
    if (!r.ok) throw new ApiError(r.status, "Could not read auth configuration");
    return r.json() as Promise<AuthConfig>;
  });
  return configPromise;
}

/**
 * Cognito's own HTTPS API, called directly from the browser.
 *
 * No SDK and no AWS credentials: the app client has no secret, which is what
 * makes it usable from a public page. amazon-cognito-identity-js would add
 * ~150KB to do these two calls.
 */
async function cognito<T>(action: string, body: unknown): Promise<T> {
  const { region, endpoint } = await authConfig();
  const res = await fetch(endpoint ? `${endpoint.replace(/\/$/, "")}/` : `https://cognito-idp.${region}.amazonaws.com/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-amz-json-1.1",
      "X-Amz-Target": `AWSCognitoIdentityProviderService.${action}`,
    },
    body: JSON.stringify(body),
  });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) {
    // Cognito puts a readable sentence in __type/message; the status alone is 400
    // for every failure including a wrong password.
    throw new ApiError(res.status, j.message || j.__type || "Sign-in failed");
  }
  return j as T;
}

interface AuthResult {
  AuthenticationResult?: { IdToken: string; RefreshToken?: string; ExpiresIn: number };
  ChallengeName?: string;
}

/** Exchange email and password for tokens. Throws ApiError with Cognito's message. */
export async function cognitoSignIn(email: string, password: string): Promise<void> {
  const { client_id } = await authConfig();
  const r = await cognito<AuthResult>("InitiateAuth", {
    AuthFlow: "USER_PASSWORD_AUTH",
    ClientId: client_id,
    AuthParameters: { USERNAME: email, PASSWORD: password },
  });
  if (!r.AuthenticationResult) {
    // NEW_PASSWORD_REQUIRED and the MFA challenges land here. Administrators set
    // passwords as permanent, so this should not occur; saying so plainly beats
    // a blank screen if it ever does.
    throw new ApiError(400, `This account needs to complete "${r.ChallengeName}", which this app cannot do yet`);
  }
  tokenStore.set(r.AuthenticationResult.IdToken);
  if (r.AuthenticationResult.RefreshToken) tokenStore.setRefresh(r.AuthenticationResult.RefreshToken);
}

/** Trade the refresh token for a fresh id token. False when it is gone or rejected. */
export async function cognitoRefresh(): Promise<boolean> {
  const rt = tokenStore.getRefresh();
  if (!rt) return false;
  try {
    const { client_id } = await authConfig();
    const r = await cognito<AuthResult>("InitiateAuth", {
      AuthFlow: "REFRESH_TOKEN_AUTH",
      ClientId: client_id,
      AuthParameters: { REFRESH_TOKEN: rt },
    });
    if (!r.AuthenticationResult?.IdToken) return false;
    tokenStore.set(r.AuthenticationResult.IdToken);
    return true;
  } catch {
    // A disabled or deleted user fails here. This is the point at which
    // revocation actually bites, so failing means logging out.
    return false;
  }
}

export async function cognitoSignOut(): Promise<void> {
  const rt = tokenStore.getRefresh();
  tokenStore.clear();
  if (!rt) return;
  try {
    const { client_id } = await authConfig();
    // Invalidates the refresh token server-side, so clearing localStorage is not
    // the only thing standing between a shared browser and the next person.
    await cognito("RevokeToken", { ClientId: client_id, Token: rt });
  } catch { /* best effort: the local tokens are already gone */ }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

function headers(json = true): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  const t = tokenStore.get();
  if (t) h.Authorization = `Bearer ${t}`;
  return h;
}

function loggedOut(): void {
  tokenStore.clear();
  window.dispatchEvent(new Event("assistant:unauthorized"));
}

async function fail(res: Response): Promise<never> {
  if (res.status === 401) loggedOut();
  let detail = res.statusText;
  try {
    const j = await res.json();
    detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
  } catch { /* not JSON */ }
  throw new ApiError(res.status, detail);
}

export async function api<T = void>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const send = () => fetch(path, {
    method: opts.method ?? "GET",
    headers: headers(),
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });

  let res = await send();

  // Five-minute id tokens mean a 401 is the expected state of a session that has
  // been open for six minutes, not a sign-in problem. Refresh once and retry;
  // only a failed refresh -- which is what a disabled account produces -- ends
  // the session.
  if (res.status === 401 && (await cognitoRefresh())) {
    res = await send();
  }

  if (!res.ok) return fail(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

/**
 * Send something that is not JSON (photos, a voice note). The browser sets the
 * multipart boundary itself, so no Content-Type is given here.
 */
async function sendForm(path: string, form: FormData, signal?: AbortSignal): Promise<Response> {
  const send = () => fetch(path, { method: "POST", headers: headers(false), body: form, signal });
  let res = await send();
  if (res.status === 401 && (await cognitoRefresh())) res = await send();
  return res;
}

/** Store and transcribe a voice note. The transcript goes back into the composer to be checked before sending. */
export async function uploadAudio(sessionId: string, audio: Blob, language: string):
    Promise<{ audio_id: string; transcript: string; language: string }> {
  const form = new FormData();
  form.append("file", audio, audio.type.includes("mp4") ? "note.m4a" : "note.webm");
  form.append("language", language);
  const res = await sendForm(`/api/sessions/${sessionId}/audio`, form);
  if (!res.ok) return fail(res);
  return res.json();
}

/**
 * An authenticated image as an object URL. `<img src>` cannot carry the bearer
 * token, so the bytes are fetched and handed to the tag as a blob. The caller
 * revokes the URL when the image goes away.
 */
export async function authedObjectUrl(path: string): Promise<string> {
  const send = () => fetch(path, { headers: headers(false) });
  let res = await send();
  if (res.status === 401 && (await cognitoRefresh())) res = await send();
  if (!res.ok) return fail(res);
  return URL.createObjectURL(await res.blob());
}

export interface OutgoingMessage { text: string; option_index?: number; images?: Blob[]; audio_id?: string }

/** POST a chat message and consume the SSE stream (fetch, not EventSource: we need the auth header). */
export async function streamMessage(
  sessionId: string,
  body: OutgoingMessage,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const path = `/api/sessions/${sessionId}/messages`;
  const multipart = (body.images?.length ?? 0) > 0 || !!body.audio_id;
  const send = () => {
    if (!multipart) {
      return fetch(path, { method: "POST", headers: headers(), signal,
                           body: JSON.stringify({ text: body.text, option_index: body.option_index }) });
    }
    const form = new FormData();
    form.append("text", body.text);
    body.images?.forEach((b, i) => form.append("images", b, `photo-${i + 1}.jpg`));
    if (body.audio_id) form.append("audio_id", body.audio_id);
    return fetch(path, { method: "POST", headers: headers(false), body: form, signal });
  };
  let res = await send();
  if (res.status === 401 && (await cognitoRefresh())) res = await send();
  if (!res.ok || !res.body) return fail(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent({ event, data: JSON.parse(data.join("\n")) } as StreamEvent);
    }
  }
}
