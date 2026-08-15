/// <reference types="vite/client" />

/** Build-time configuration injected by Vite (see api.ts). */
interface ImportMetaEnv {
  /** API origin. "" in a deployed build, meaning "same origin as this page". */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
