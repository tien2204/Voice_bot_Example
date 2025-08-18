kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl apply -f cluster-issuer.yaml
kubectl apply -f wildcard-issuer.yaml

echo "Waiting for cert-manager webhook to be ready..."
if ! kubectl wait --for=condition=available deployment/cert-manager-webhook -n cert-manager --timeout=180s; then
  echo "Error: deployment/cert-manager-webhook did not become Ready within 180s." >&2
  exit 1
fi


echo "Waiting for Certificate 'sehrmude-wildcard' to be issued and Secret 'sehrmude-wildcard-tls' to be created in namespace 'default'..."
if ! kubectl wait --for=condition=Ready certificate/sehrmude-wildcard -n default --timeout=300s; then
  echo "Error: Certificate 'sehrmude-wildcard' did not become Ready within 300s." >&2
  exit 1
fi

# echo "Adding Helm repositories..."
helm repo add metallb https://metallb.github.io/metallb
helm repo add livekit https://helm.livekit.io
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update



echo "Installing/Updating Redis..."
if ! helm upgrade --install redis bitnami/redis \
  --namespace default \
  --set architecture=standalone \
  --set auth.enabled=false \
  --set persistence.enabled=false; then
  echo "Error: Helm upgrade/install for Redis failed." >&2
  exit 1
fi

echo "Waiting for Redis to be ready..."
if ! kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=redis,app.kubernetes.io/instance=redis,app.kubernetes.io/component=master \
  -n default --timeout=300s; then
  echo "Error: Redis master pod did not become Ready within 300s." >&2
  exit 1
fi

helm upgrade --install metallb metallb/metallb -f metallb-value.yaml -n metallb-system --create-namespace 
kubectl apply -f metallb-config.yaml

echo "Installing/Updating LiveKit server..."
helm upgrade --install livekit livekit/livekit-server --namespace default --values value.yaml
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.publishService.enabled=true
  
  # --set controller.service.type=NodePort \
  # --set controller.service.nodePorts.http=32080 \
  # --set controller.service.nodePorts.https=32443 \
  # --set controller.config.log-format-upstream='$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" "$request_length"'
kubectl apply -f ingress.yaml