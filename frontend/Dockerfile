# ─────────────────────────────────────────────────────────────
# Frontend image — Node 20 (Vite dev server)
# In docker-compose the source is mounted and `npm install` runs at
# startup, but this Dockerfile lets the image build & run standalone too.
# ─────────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
