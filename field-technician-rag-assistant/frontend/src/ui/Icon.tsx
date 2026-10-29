/**
 * Inline SVG icons.
 *
 * Inline rather than an icon package: this needs nine glyphs, and a library costs
 * tens of kilobytes for them. They inherit `currentColor`, so a themed parent
 * colours them with no extra rule.
 *
 * `aria-hidden` on every one. The icon is never the accessible name — the button
 * around it carries an aria-label, so a screen reader announcing the glyph too
 * would read everything twice.
 */
type Props = { className?: string };

const box = {
  // An intrinsic size, so an icon is correct wherever it is used. Without it an
  // SVG carrying only a viewBox falls back to the replaced-element default and
  // fills whatever box it is in -- which is how the new-chat plus ended up
  // enormous: only `.ico svg` set a size, and that button is not an `.ico`.
  // CSS still overrides this where a different size is wanted.
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const IconMenu = (p: Props) => (
  <svg {...box} {...p}><path d="M4 6h16M4 12h16M4 18h16" /></svg>
);

export const IconClose = (p: Props) => (
  <svg {...box} {...p}><path d="M6 6l12 12M18 6L6 18" /></svg>
);

export const IconPlus = (p: Props) => (
  <svg {...box} {...p}><path d="M12 5v14M5 12h14" /></svg>
);

/** Sun for "currently light". Replaces the gear, which suggested settings. */
export const IconSun = (p: Props) => (
  <svg {...box} {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
  </svg>
);

/** Moon for "currently dark". */
export const IconMoon = (p: Props) => (
  <svg {...box} {...p}><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5z" /></svg>
);

export const IconPin = ({ filled, ...p }: Props & { filled?: boolean }) => (
  <svg {...box} {...p} fill={filled ? "currentColor" : "none"}>
    <path d="M9 4h6l-.7 5.2 3.2 3.3H6.5l3.2-3.3L9 4z" />
    <path d="M12 12.5V20" />
  </svg>
);

export const IconSend = (p: Props) => (
  <svg {...box} {...p} strokeWidth={2}><path d="M5 12h13M12 5l7 7-7 7" /></svg>
);

export const IconSliders = (p: Props) => (
  <svg {...box} {...p}>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" /><circle cx="10" cy="17" r="2" />
  </svg>
);

/** IEC 5009 power symbol: a broken ring with a line through the gap. The same
 *  mark as the on/off switch on the machines these technicians service. */
export const IconPower = (p: Props) => (
  <svg {...box} {...p}>
    <path d="M18.36 6.64a9 9 0 1 1-12.72 0" />
    <path d="M12 2v10" />
  </svg>
);

export const IconChat = (p: Props) => (
  <svg {...box} {...p}><path d="M4 5h16v11H9l-5 4V5z" /></svg>
);

/** A filled square — the universal "stop", and unmistakable next to the arrow. */
export const IconStop = (p: Props) => (
  <svg {...box} {...p}><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none" /></svg>
);

/** Two offset rounded squares — the copy mark people already recognise. */
export const IconCopy = (p: Props) => (
  <svg {...box} {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2.5" />
    <path d="M15.5 9V6.5A2.5 2.5 0 0 0 13 4H6.5A2.5 2.5 0 0 0 4 6.5V13a2.5 2.5 0 0 0 2.5 2.5H9" />
  </svg>
);

export const IconCheck = (p: Props) => (
  <svg {...box} {...p} strokeWidth={2.2}><path d="M4 12.5l5 5L20 6.5" /></svg>
);

export const IconPaperclip = (p: Props) => (
  <svg {...box} {...p}><path d="M21 11.5l-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l8.9-8.9a3.7 3.7 0 0 1 5.2 5.2l-8.9 8.9a1.8 1.8 0 0 1-2.6-2.6l8.2-8.2" /></svg>
);

export const IconMic = (p: Props) => (
  <svg {...box} {...p}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" /></svg>
);
