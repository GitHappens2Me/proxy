import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from .cashu import _initialize_wallet

admin_router = APIRouter(prefix="/admin")


def login_form() -> str:
    return """<!DOCTYPE html>
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                input[type="password"] {
                    padding: 8px;
                }
                button {
                    padding: 8px;
                    cursor: pointer;
                }
            </style>
            <script>
                function handleSubmit(e) {
                    e.preventDefault();
                    const password = document.getElementById('password').value;
                    document.cookie = `admin_password=${password}; path=/; max-age=86400`;
                    window.location.reload();
                }
            </script>
        </head>
        <body>
            <form onsubmit="handleSubmit(event)">
                <input type="password" id="password" placeholder="Admin Password" required>
                <button type="submit">Login</button>
            </form>
        </body>
    </html>
    """


def info(content: str) -> str:
    return f"""<!DOCTYPE html>
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
            </style>
        </head>
        <body>
            <div style="text-align: center;">
                {content}
            </div>
        </body>
    </html>
    """

def admin_auth() -> str:
    if os.getenv("ADMIN_PASSWORD", "") == "":
        return info("Please set a secure ADMIN_PASSWORD= in your ENV variables.")
    else:
        return login_form()


from sqlmodel import select
from .db import ApiKey, create_session

async def dashboard(request: Request) -> str:

    # fetch cashu / api-key data from database
    async with create_session() as session:
        result = await session.exec(select(ApiKey))
        api_keys = result.all()
    api_keys_table_rows = "".join(
        f"<tr><td>{key.hashed_key}</td><td>{key.balance}</td><td>{key.refund_address}</td><td>{key.total_spent}</td><td>{key.total_requests}</td></tr>"
        for key in api_keys
    )

    # Fetch balance from cashu
    wallet = await _initialize_wallet()
    await wallet.load_proofs(reload=True)  # <--- ADD THIS
    current_balance = wallet.balance  # Not awaited, assuming it's a property

    wallet = await _initialize_wallet("https://mint.minibits.cash/Bitcoin")
    await wallet.load_proofs(reload=True)  # <--- ADD THIS
    minibits_balance = wallet.balance  # Not awaited, assuming it's a property


    return f"""<!DOCTYPE html>
    <html>
        <head>
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }}
            </style>
        </head>
        <body>
            <h1>Admin Dashboard</h1>
            <h2>Current Cashu Balance (including user balances)</h2>
            <p>{current_balance} sats ({minibits_balance})</p>
            <h2>User's API Keys</h2>
            <table>
                <tr>
                    <th>Hashed Key</th>
                    <th>Balance (mSats)</th>
                    <th>Refund Address</th>
                    <th>Total Spent(mSats)</th>
                    <th>Total Requests</th>
                </tr>
                {api_keys_table_rows}
            </table>
        </body>
    </html>
    """

@admin_router.get("/", response_class=HTMLResponse)
async def admin(request: Request):
    admin_cookie = request.cookies.get("admin_password")
    if admin_cookie and admin_cookie == os.getenv("ADMIN_PASSWORD"):
        return await dashboard(request)
    return admin_auth()

