import secrets

# Generate a JWT secret key (e.g., 32 bytes, URL-safe base64 encoded)
jwt_secret_key = secrets.token_urlsafe(32)
print(f"New JWT Secret Key: {jwt_secret_key}")

# Generate a WS token (e.g., 16 bytes, URL-safe base64 encoded)
ws_token = secrets.token_urlsafe(16)
print(f"New WS Token: {ws_token}")
