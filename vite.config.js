import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

export default defineConfig({
  plugins: [glsl()],
  server: {
    open: true, // automatically open the browser
  },
});