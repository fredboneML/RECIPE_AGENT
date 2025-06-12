# app/backend/ai-analyzer/main.py
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Add after /api/login endpoint


@app.post("/api/refresh-token")
async def refresh_token(request: Request, db: Session = Depends(get_db)):
    try:
        auth_header = request.headers.get('authorization')
        if not auth_header or not auth_header.lower().startswith('bearer '):
            raise HTTPException(
                status_code=401, detail="Missing or invalid authorization header")
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            tenant_code = payload.get("tenant_code")
            role = payload.get("role")
            if not username or not tenant_code or not role:
                raise HTTPException(
                    status_code=401, detail="Invalid token payload")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Optionally, check user still exists and is active
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        # Issue new token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_token = create_access_token(
            data={
                "sub": user.username,
                "tenant_code": user.tenant_code,
                "role": user.role
            },
            expires_delta=access_token_expires
        )
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {e}")
        raise HTTPException(status_code=401, detail="Could not refresh token")
