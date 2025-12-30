FROM nginx:stable
# Paths relative to repo root (Railway builds from root)
COPY --chown=nginx:nginx scripts/extension/releases /usr/share/nginx/html
COPY --chown=nginx:nginx scripts/extension/dataset /usr/share/nginx/html/dataset
COPY scripts/extension/nginx.conf /etc/nginx/nginx.conf
RUN rm -f /usr/share/nginx/html/index.html /usr/share/nginx/html/50x.html

