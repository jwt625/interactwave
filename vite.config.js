import { defineConfig } from 'vite';
import glsl from 'vite-plugin-glsl';

export default defineConfig({
  plugins: [glsl()],
  base: './',
  build: {
    target: 'ES2022',
  },
  resolve: {
    extensions: ['.js', '.frag', '.vert'],
  }
});