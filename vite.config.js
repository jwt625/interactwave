import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

export default defineConfig({
  plugins: [glsl()],
  root: './',
  build: {
    target: 'ES2022',
  },
});