import os
import httpx

from cashu.core.base import Token  # type: ignore
from cashu.wallet.wallet import Wallet  # type: ignore
from cashu.wallet.helpers import deserialize_token_from_string, receive  # type: ignore
from sqlmodel import select, func, col
from .db import ApiKey, AsyncSession

RECEIVE_LN_ADDRESS = os.environ["RECEIVE_LN_ADDRESS"]
MINT = os.environ.get("MINT", "https://mint.minibits.cash/Bitcoin")
MINIMUM_PAYOUT = int(os.environ.get("MINIMUM_PAYOUT", 100))
DEV_LN_ADDRESS = "routstr@minibits.cash"
DEVS_DONATION_RATE = 0.021  # 2.1%
WALLET = None

#TODO
# This causes problems when users send tokens from other mints
# WALLET is already set so it returns the specified wallet, but this wallet does not know the keyset of the token 
# Define the global dictionary to hold wallets (fixes "WALLETS doesn't exist")
WALLETS: dict[str, Wallet] = {}  # Maps mint URLs to initialized wallets

# ... (other imports and constants remain the same) ...

async def _initialize_wallet(mint_url: str | None = None) -> Wallet:
    """Initializes and loads a Cashu wallet with isolated storage via unique names."""
    if mint_url is None:
        mint_url = MINT  # Use the default MINT URL if none provided
    
    # Create a unique wallet identifier based on mint URL
    mint_identifier = (
        mint_url.replace("://", "_").replace(".", "_").replace("/", "_").lower()
    )
    wallet_name = f"cashu_wallet_{mint_identifier}"
    
    # Check if wallet already exists in the dictionary
    if mint_url in WALLETS:
        print(f"found wallet: {mint_url}")
        return WALLETS[mint_url]
    
    print(f"Initializing wallet for mint: {mint_url}")
    
    #TODO same direction but different names!
    # Initialize the wallet with unique name and shared base directory
    wallet = await Wallet.with_db(
        mint_url,
        db=f"./{wallet_name}",  # Full path combines base + unique name
        name=wallet_name,       # Unique identifier
        load_all_keysets=True,  # Ensure all keysets are loaded
        unit="sat",
    )
    
    # Load mint info and keysets
    await wallet.load_mint_info()
    await wallet.load_mint_keysets()
    
    # Activate keyset if none is active
    if not hasattr(wallet, "keyset_id") or wallet.keyset_id is None:
        await wallet.activate_keyset()
    
    # Load existing proofs
    await wallet.load_proofs(reload=True)
    print(f"{wallet.balance=}")
    # Store the initialized wallet in the global dict
    WALLETS[mint_url] = wallet
    return wallet

async def _handle_token_receive(wallet: Wallet, token_obj: Token) -> int:
    """Receives a token and returns the amount received."""
    initial_balance = wallet.available_balance
    print("receive")
    await receive(wallet, token_obj)
    print("load_proofs")
    await wallet.load_proofs(reload=True)
    final_balance = wallet.available_balance
    amount_received = final_balance - initial_balance

    if amount_received <= 0:
        raise ValueError("Token contained no value.")
    return amount_received * 1000


async def _get_lnurl_invoice(callback_url: str, amount_msat: int) -> tuple[str, dict]:
    """Requests an invoice from the LNURL callback URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            callback_url,
            params={"amount": amount_msat},
            follow_redirects=True,
        )
        response.raise_for_status()  # Raise exception for non-2xx status codes
    invoice_data = response.json()
    if "pr" not in invoice_data:
        raise ValueError(f"Invalid LNURL invoice response: {invoice_data}")
    return invoice_data["pr"], invoice_data


async def _pay_invoice_with_cashu(
    wallet: Wallet, bolt11_invoice: str, amount_to_send_msat: int
) -> int:
    """Pays a BOLT11 invoice using Cashu proofs via melt."""

    amount_to_send_msat = amount_to_send_msat // 1000 #???????????TDOODO
    quote = await wallet.melt_quote(bolt11_invoice, amount_to_send_msat)

    proofs_to_melt, _ = await wallet.select_to_send(
        wallet.proofs, quote.amount + quote.fee_reserve
    )
    #print(f"Proofs to melt: {proofs_to_melt}")

    _ = await wallet.melt(
        proofs_to_melt, bolt11_invoice, quote.fee_reserve, quote.quote
    )
    print(f"Payed {quote.amount}")
    return quote.amount


async def pay_out_with_new_session() -> None:
    """
    Wrapper for pay_out that creates its own database session.
    This prevents database connection conflicts when called as a background task.
    """
    from .db import create_session
    
    try:
        async with create_session() as session:
            await pay_out(session)
    except Exception as e:
        print(f"Error in pay_out_with_new_session: {e}")


async def pay_out(session: AsyncSession) -> None:
    """
    Calculates the pay-out amount based on the spent balance, profit, and donation rate.
    """
    try:
        balance = (
            await session.exec(
                select(func.sum(col(ApiKey.balance))).where(ApiKey.balance > 0)
            )
        ).one()
        if balance is None or balance == 0:
            # No balance to pay out - this is OK, not an error
            return
        
        user_balance_sats = balance // 1000  # Convert msats to sats
        wallet = await _initialize_wallet()
        wallet_balance_sats = wallet.available_balance  # Already in sats

        # Handle edge cases more gracefully
        if wallet_balance_sats < user_balance_sats:
            print(f"Warning: Wallet balance ({wallet_balance_sats} sats) is less than user balance ({user_balance_sats} sats). Skipping payout.")
            return

        if (revenue := wallet_balance_sats - user_balance_sats) <= MINIMUM_PAYOUT:
            # Not enough revenue yet - this is OK
            return

        devs_donation = int(revenue * DEVS_DONATION_RATE)
        owners_draw = revenue - devs_donation

        # Send payouts
        print(f"Sending {owners_draw} sats to {RECEIVE_LN_ADDRESS}")
        await send_to_lnurl(wallet, RECEIVE_LN_ADDRESS, owners_draw * 1000)  # Convert to msats
        print(f"Sending {devs_donation} sats to {DEV_LN_ADDRESS}")
        await send_to_lnurl(
            wallet,
            DEV_LN_ADDRESS,
            devs_donation * 1000,  # Convert to msats
        )

    except Exception as e:
        # Log the error but don't crash - payouts can be retried later
        print(f"Error in pay_out: {e}")

async def credit_balance(cashu_token: str, key: ApiKey, session: AsyncSession) -> int:
    # 1. Deserialize token and validate
    try:
        token_obj: Token = deserialize_token_from_string(cashu_token)
        token_mint = token_obj.mint
        print(f"[DEBUG] Received token from mint: {token_mint}")
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize token: {e}")

    # 2. Check if token is from foreign mint
    if token_mint != MINT:
        print("=== CROSS-MINT TRANSACTION ===")

        # Step 1: Initialize foreign wallet
        try:
            foreign_wallet = await _initialize_wallet(token_mint)
            print(f"[DEBUG] Foreign wallet initialized for {token_mint}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize foreign wallet: {e}")

        # Step 2: Receive tokens into foreign wallet
        try:
            foreign_amount_msats = await _handle_token_receive(foreign_wallet, token_obj)
            assert foreign_amount_msats > 0, "Received zero tokens"
            foreign_amount_sats = foreign_amount_msats // 1000
            print(f"[DEBUG] Received {foreign_amount_sats} SAT from foreign mint")
        except Exception as e:
            raise RuntimeError(f"Token receive failed: {e}")

        # Step 3: Calculate fees and net amount
        fee_percent = 0.01
        net_amount_sats = int(foreign_amount_sats * (1 - fee_percent))
        print(f"[DEBUG] Net amount after fees: {net_amount_sats} SAT")

        # Step 4: Create invoice from original mint
        try:
            original_wallet = await _initialize_wallet(MINT)
            print(f"[DEBUG] Original wallet initialized for {MINT}")
            mint_quote = await original_wallet.request_mint(
                net_amount_sats, 
                memo="Cross-mint conversion"
            )
            print(f"[DEBUG] Generated mint_quote: {mint_quote}")
            invoice = mint_quote.request
            print(f"[DEBUG] Generated invoice: {invoice}")
        except Exception as e:
            raise RuntimeError(f"Invoice creation failed: {e}")

        # Step 5: Pay invoice using foreign tokens
        try:
            await _pay_invoice_with_cashu(
                foreign_wallet, 
                invoice, 
                net_amount_sats * 1000  # Convert sats to msats
            )
            print(f"[DEBUG] Invoice paid successfully")
        except Exception as e:
            raise RuntimeError(f"Invoice payment failed: {e}")

        # Final step: Credit user's balance
        key.balance += net_amount_sats * 1000
        session.add(key)
        await session.commit()
        print(f"[DEBUG] Balance credited: {net_amount_sats * 1000} msats")
        return net_amount_sats * 1000

    else:
        print("=== SAME-MINT TRANSACTION ===")

        # Handle same-mint case
        try:
            wallet = await _initialize_wallet(token_obj.mint)
            amount_msats = await _handle_token_receive(wallet, token_obj)
            assert amount_msats > 0, "Received zero tokens"
            print(f"[DEBUG] Received {amount_msats} msats from same mint")
        except Exception as e:
            raise RuntimeError(f"Token receive failed: {e}")

        key.balance += amount_msats
        session.add(key)
        await session.commit()
        print(f"[DEBUG] Balance credited: {amount_msats} msats")
        return amount_msats

async def refund_balance(amount: int, key: ApiKey, session: AsyncSession) -> int:
    wallet = await _initialize_wallet()
    if key.balance < amount:
        raise ValueError("Insufficient balance.")
    if amount <= 0:
        amount = key.balance
    key.balance -= amount
    session.add(key)
    await session.commit()
    if key.refund_address is None:
        raise ValueError("Refund address not set.")
    return await send_to_lnurl(wallet, key.refund_address, amount_msat=amount) #Todo: Check possible msats / sats conversion error?


async def create_token(
    amount_msats: int, mint: str = MINT
) -> str:
    wallet = await _initialize_wallet(mint)
    balance = wallet.available_balance
    amount_sats = amount_msats // 1000
    if balance < amount_sats:
        raise ValueError("Insufficient balance on mint.")
    print(balance, amount_sats)
    if balance > amount_sats:
        print("splitting")
        _, send_proofs = await wallet.split(wallet.proofs, amount_sats)
    else:
        print("no splitting")
        send_proofs = wallet.proofs
    token = await wallet._make_tokenv4(send_proofs)
    return token.serialize()


async def redeem(cashu_token: str, lnurl: str) -> int:
    """
    Redeems a Cashu token and sends the amount to an LNURL address.

    Args:
        cashu_token: The Cashu token string (starting with "cashuA...").
        lnurl: The LNURL string (can be bech32, user@host, or direct URL).

    Returns:
        The amount in millisatoshis that was successfully sent.

    Raises:
        Exception: If any step of the process fails (token receive, LNURL fetch, invoice payment).
    """
    token_obj: Token = deserialize_token_from_string(cashu_token)
    wallet: Wallet = await _initialize_wallet(token_obj.mint)

    amount_received = await _handle_token_receive(wallet, token_obj)

    # if USE_BALANCE_ON_INVALID_TOKEN:
    #     amount_received = wallet.available_balance

    return await send_to_lnurl(wallet, lnurl, amount_received)


async def send_to_lnurl(wallet: Wallet, lnurl: str, amount_msat: int) -> int:
    """
    Sends funds from a Cashu wallet to an LNURL address.

    Args:
        wallet: The initialized Cashu wallet with available balance.
        lnurl: The LNURL string (can be bech32, user@host, or direct URL).
        amount_msat: The amount in millisatoshis to send.

    Returns:
        The amount in millisatoshis that was successfully sent.

    Raises:
        ValueError: If amount is outside LNURL limits or other validation errors.
        Exception: If LNURL fetch or invoice payment fails.
    """
    print(f"Sending {amount_msat / 1000} sat to {lnurl}")
    callback_url, min_sendable, max_sendable = await get_lnurl_data(lnurl)

    if not (min_sendable <= amount_msat <= max_sendable):
        raise ValueError(
            f"Amount {amount_msat / 1000} sat is outside LNURL limits "
            f"({min_sendable / 1000} - {max_sendable / 1000} sat)."
        )
    # subtract estimated fees
    amount_to_send = amount_msat - int(max(2000, amount_msat * 0.01))


    print(f"trying to pay {amount_to_send} msats to {lnurl}. Available balance: {wallet.balance}", flush=True)
    # Note: We pass amount_msat directly. The actual amount paid might be adjusted
    # slightly by the melt quote based on the invoice details.
    bolt11_invoice, _ = await _get_lnurl_invoice(callback_url, amount_to_send)

    # Conversion to Sats (/ 1000 necessary for cashu payments)
    amount_paid = await _pay_invoice_with_cashu(wallet, bolt11_invoice, amount_to_send / 1000)

    print(f"{amount_paid} sats paid to lnurl", flush=True)

    print(f"Amount paid: {amount_paid / 1000} sat")
    return amount_paid


async def get_lnurl_data(lnurl: str) -> tuple[str, int, int]:
    """
    Fetches LNURL payRequest data (callback URL, min/max sendable amounts).

    Handles lightning:, user@host, bech32 lnurl, and direct HTTPS URL formats.
    """
    url: str
    if lnurl.startswith("lightning:"):
        lnurl = lnurl[10:]

    if "@" in lnurl and len(lnurl.split("@")) == 2:
        user, host = lnurl.split("@")
        url = f"https://{host}/.well-known/lnurlp/{user}"
    elif lnurl.lower().startswith("lnurl"):
        try:
            # Optional import for environments where bech32 might not be present initially
            from bech32 import bech32_decode, convertbits  # type: ignore

            hrp, data = bech32_decode(lnurl)
            if data is None:
                raise ValueError("Invalid bech32 data in LNURL")
            decoded_data = convertbits(data, 5, 8, False)
            if decoded_data is None:
                raise ValueError("Failed to convert LNURL bits")
            url = bytes(decoded_data).decode("utf-8")
        except ImportError:
            raise ImportError("bech32 library is required for LNURL bech32 decoding.")
        except Exception as e:
            raise ValueError(f"Failed to decode LNURL: {e}") from e
    else:
        # Assume it's a direct URL
        if not lnurl.startswith("https://"):
            # Basic check, could be improved
            raise ValueError("Direct LNURL must use HTTPS")
        url = lnurl

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
        lnurl_data: dict = response.json()

    if lnurl_data.get("tag") != "payRequest" or not isinstance(
        lnurl_data.get("callback"), str
    ):
        raise ValueError(f"Invalid LNURL payRequest data: {lnurl_data}")

    callback_url: str = lnurl_data["callback"]
    # LNURL spec defaults (in millisatoshis)
    min_sendable: int = lnurl_data.get("minSendable", 1000)
    max_sendable: int = lnurl_data.get("maxSendable", 1000000000)  # Default 1000 BTC

    return callback_url, min_sendable, max_sendable


if __name__ == "__main__":
    import asyncio

    # Example usage: Replace with your actual LNURL and Token
    lnurl = "user@walletofsatoshi.com"  # Replace
    # A potentially spent token for testing fallback logic
    cashu_token = "cashuBpGF0gaJhaUg..."

    # Example: Set USE_BALANCE_ON_INVALID_TOKEN = False to test non-fallback behavior
    # USE_BALANCE_ON_INVALID_TOKEN = True

    async def main() -> None:
        # Removed try-except block, script will crash on error
        print(f"Attempting to redeem token and pay LNURL: {lnurl}")
        amount_sent = await redeem(cashu_token, lnurl)
        print(f"✅ Successfully sent {amount_sent / 1000} sat ({amount_sent} msat).")

    # Removed try-except block for KeyboardInterrupt
    asyncio.run(main())
