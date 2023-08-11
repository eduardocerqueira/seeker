#date: 2023-08-11T16:48:30Z
#url: https://api.github.com/gists/b0e5624dd8b35e999e32301350aaf177
#owner: https://api.github.com/users/babldev

# Alpine image is smaller but Prisma is broken on M1 https://github.com/prisma/prisma/issues/8478
# FROM node:18-alpine AS builder
# RUN apk update
# RUN apk add --no-cache openssl
FROM node:18-slim AS runner
RUN apt-get update \
    && apt-get install -y openssl

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Alpine image is smaller but Prisma is broken on M1 https://github.com/prisma/prisma/issues/8478
# FROM node:18-alpine AS builder
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
# RUN apk update
# RUN apk add --no-cache libc6-compat openssl
FROM node:18-slim AS builder
RUN apt-get update \
    && apt-get install -y openssl
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED 1

COPY \
    jest.config.ts \
    next-env.d.ts \
    next.config.js \
    package-lock.json \
    package.json \
    tsconfig.json \
    ./
COPY src/ src/
COPY prisma/ prisma/
COPY types/ types/

RUN npm ci
RUN npx prisma generate
RUN npm run build

#
# Production image, copy all the files and run next
#
FROM runner
WORKDIR /app

# You only need to copy next.config.js if you are NOT using the default configuration
COPY --from=builder /app/next.config.js ./
# COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY public/ /app/public/

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
