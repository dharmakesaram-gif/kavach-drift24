/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#070b14",
        panel: "#0d1424",
        edge: "#1c2a44",
        accent: "#7dd3fc",
        muted: "#8aa0c0",
        accept: "#4ade80",
        review: "#fbbf24",
        reject: "#f87171"
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
