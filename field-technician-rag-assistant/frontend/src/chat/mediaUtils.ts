/**
 * Photos are shrunk in the browser before upload: a phone photo is 3-8 MB and the
 * shop floor's signal is often poor. The server re-encodes whatever arrives
 * anyway (and strips EXIF), so this is about speed, not trust.
 */
const MAX_EDGE = 2048;

export async function shrinkPhoto(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(() => null);
  if (!bitmap) return file; // unusual format: let the server decide
  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b ?? file), "image/jpeg", 0.85));
}

/** The best recording format this browser offers that Transcribe accepts (WebM/Opus, or MP4/AAC on Safari). */
export function recorderMimeType(): string | undefined {
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return options.find((t) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t));
}
