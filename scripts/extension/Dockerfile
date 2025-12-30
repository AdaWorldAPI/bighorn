# Use the pre-built extension repo image from GHCR
# Same image as in compose.yaml
FROM ghcr.io/kuzudb/extension-repo:latest

# Healthcheck for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

EXPOSE 80

