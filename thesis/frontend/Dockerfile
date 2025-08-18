FROM node:20-alpine AS deps
RUN npm install -g pnpm@9.15.9
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY src/package.json src/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile
FROM node:20-alpine AS builder
RUN npm install -g pnpm@9.15.9
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY src/. .
COPY .env .env
RUN pnpm build
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
RUN npm install -g pnpm@9.15.9
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/pnpm-lock.yaml* ./
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/public ./public 
COPY --from=builder /app/node_modules ./node_modules
COPY .env .env
USER nextjs
EXPOSE 3000
CMD ["pnpm", "start"]