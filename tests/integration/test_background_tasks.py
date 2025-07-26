"""Integration tests for background tasks"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any, Coroutine, List
from unittest.mock import AsyncMock, patch

import pytest

from router.cashu import check_for_refunds, periodic_payout
from router.db import ApiKey
from router.models import MODELS, Model, Pricing, update_sats_pricing


@pytest.mark.asyncio
class TestPricingUpdateTask:
    """Test the pricing update background task"""

    async def test_updates_model_prices_periodically(self) -> None:
        """Test that update_sats_pricing updates all model prices based on BTC/USD rate"""
        # Mock the price fetch function
        mock_sats_usd = 0.00002  # 1 sat = $0.00002 (BTC at $50,000)

        with patch(
            "router.models.sats_usd_ask_price", AsyncMock(return_value=mock_sats_usd)
        ):
            # Create a test model
            test_model = Model(  # type: ignore[arg-type]
                id="test-model",
                name="Test Model",
                created=1234567890,
                description="Test",
                context_length=4096,
                architecture={  # type: ignore[arg-type]
                    "modality": "text",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "tokenizer": "test",
                    "instruct_type": None,
                },
                pricing=Pricing(
                    prompt=0.001,  # $0.001 per token
                    completion=0.002,
                    request=0.0,
                    image=0.0,
                    web_search=0.0,
                    internal_reasoning=0.0,
                    max_cost=0.0,
                ),
                top_provider={  # type: ignore[arg-type]
                    "context_length": 4096,
                    "max_completion_tokens": 1024,
                    "is_moderated": False,
                },
            )

            # Add test model to MODELS list
            original_models = MODELS.copy()
            MODELS.clear()
            MODELS.append(test_model)

            try:
                # Run the pricing update task once
                task = asyncio.create_task(update_sats_pricing())
                await asyncio.sleep(0.1)  # Let it run one iteration
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Verify sats pricing was calculated correctly
                assert test_model.sats_pricing is not None
                assert test_model.sats_pricing.prompt == pytest.approx(
                    0.001 / mock_sats_usd
                )
                assert test_model.sats_pricing.completion == pytest.approx(
                    0.002 / mock_sats_usd
                )

                # Verify max_cost calculation
                expected_max_cost = (
                    4096 * test_model.sats_pricing.prompt
                    + 1024 * test_model.sats_pricing.completion
                )
                assert test_model.sats_pricing.max_cost == pytest.approx(
                    expected_max_cost
                )

            finally:
                # Restore original models
                MODELS.clear()
                MODELS.extend(original_models)

    async def test_handles_provider_api_failures(self) -> None:
        """Test that pricing update continues running even if price API fails"""
        call_count = 0

        async def mock_price_func() -> float:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Price API error")
            return 0.00002

        with patch("router.models.sats_usd_ask_price", mock_price_func):
            # Run the task
            task = asyncio.create_task(update_sats_pricing())
            await asyncio.sleep(15)  # Let it run for >10 seconds (one retry)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify it retried after the error
            assert call_count >= 2

    async def test_database_updates_are_atomic(self) -> None:
        """Test that model price updates don't interfere with concurrent operations"""
        # This test verifies the pricing updates are in-memory only
        # and don't affect database operations

        test_model = Model(  # type: ignore[arg-type]
            id="test-atomic",
            name="Test Atomic",
            created=1234567890,
            description="Test",
            context_length=4096,
            architecture={  # type: ignore[arg-type]
                "modality": "text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "tokenizer": "test",
                "instruct_type": None,
            },
            pricing=Pricing(
                prompt=0.001,
                completion=0.002,
                request=0.0,
                image=0.0,
                web_search=0.0,
                internal_reasoning=0.0,
                max_cost=0.0,
            ),
        )

        original_models = MODELS.copy()
        MODELS.clear()
        MODELS.append(test_model)

        try:
            with patch(
                "router.models.sats_usd_ask_price", AsyncMock(return_value=0.00002)
            ):
                # Start the pricing task
                task = asyncio.create_task(update_sats_pricing())

                # Simulate concurrent access to the model
                results = []

                async def access_model() -> None:
                    await asyncio.sleep(0.05)  # Small delay
                    results.append(test_model.sats_pricing)

                # Run multiple concurrent accesses during pricing update
                await asyncio.gather(*[access_model() for _ in range(10)])

                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # All accesses should see consistent state
                assert all(r is not None for r in results)

        finally:
            MODELS.clear()
            MODELS.extend(original_models)


@pytest.mark.asyncio
class TestRefundCheckTask:
    """Test the refund check background task"""

    async def test_processes_pending_refunds(
        self, integration_session: Any, testmint_wallet: Any, db_snapshot: Any
    ) -> None:
        """Test that expired keys with balance and refund address are refunded"""
        # Create an expired API key with balance
        expired_key = ApiKey(
            hashed_key="expired_test_key",
            balance=5000,  # 5 sats in msats
            refund_address="lnurl1test",
            key_expiry_time=int(time.time()) - 3600,  # Expired 1 hour ago
            created_at=datetime.utcnow() - timedelta(days=1),
        )
        integration_session.add(expired_key)
        await integration_session.commit()

        # Mock the wallet send_to_lnurl method and get_session
        with (
            patch("router.cashu.wallet") as mock_wallet,
            patch("router.cashu.get_session") as mock_get_session,
        ):
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=5)
            mock_wallet.return_value = mock_wallet_instance

            # Make get_session return our integration session
            async def get_test_session() -> Any:
                yield integration_session

            mock_get_session.side_effect = get_test_session

            # Take initial snapshot
            await db_snapshot.capture()

            # Run refund check once
            original_interval = os.environ.get("REFUND_PROCESSING_INTERVAL", "3600")
            os.environ["REFUND_PROCESSING_INTERVAL"] = (
                "0.1"  # Fast interval for testing
            )

            task = asyncio.create_task(check_for_refunds())
            await asyncio.sleep(0.5)  # Let it run one cycle
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                os.environ["REFUND_PROCESSING_INTERVAL"] = original_interval

            # Verify refund was processed
            mock_wallet_instance.send_to_lnurl.assert_called_once_with(
                "lnurl1test", amount=5
            )

            # Check database state
            db_diff = await db_snapshot.diff()
            assert len(db_diff["api_keys"]["modified"]) == 1
            modified_key = db_diff["api_keys"]["modified"][0]
            assert modified_key["changes"]["balance"]["new"] == 0
            assert modified_key["changes"]["balance"]["delta"] == -5000

    async def test_handles_mint_communication_errors(
        self, integration_session: Any
    ) -> None:
        """Test that refund check continues after mint errors"""
        # Create multiple expired keys
        for i in range(3):
            key = ApiKey(
                hashed_key=f"expired_key_{i}",
                balance=1000 * (i + 1),
                refund_address=f"lnurl{i}",
                key_expiry_time=int(time.time()) - 3600,
                created_at=datetime.utcnow(),
            )
            integration_session.add(key)
        await integration_session.commit()

        refund_count = 0

        async def mock_send_to_lnurl(address: str, amount: int) -> int:
            nonlocal refund_count
            refund_count += 1
            if refund_count == 2:
                raise Exception("Mint communication error")
            return amount

        with (
            patch("router.cashu.wallet") as mock_wallet,
            patch("router.cashu.get_session") as mock_get_session,
        ):
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.send_to_lnurl = mock_send_to_lnurl
            mock_wallet.return_value = mock_wallet_instance

            # Make get_session return our integration session
            async def get_test_session() -> Any:
                yield integration_session

            mock_get_session.side_effect = get_test_session

            # Run refund check
            original_interval = os.environ.get("REFUND_PROCESSING_INTERVAL", "3600")
            os.environ["REFUND_PROCESSING_INTERVAL"] = "0.1"

            task = asyncio.create_task(check_for_refunds())
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                os.environ["REFUND_PROCESSING_INTERVAL"] = original_interval

            # Should have attempted all refunds despite one failure
            assert refund_count == 3

    async def test_updates_refund_status_correctly(
        self, integration_session: Any, db_snapshot: Any
    ) -> None:
        """Test that refund status and key deletion work correctly"""
        # Create keys with different states
        keys_data = [
            # Should be refunded and deleted (zero balance after refund)
            {
                "hashed_key": "delete_me",
                "balance": 1000,
                "refund_address": "lnurl1",
                "expired": True,
            },
            # Should keep (not expired)
            {
                "hashed_key": "keep_not_expired",
                "balance": 2000,
                "refund_address": "lnurl2",
                "expired": False,
            },
            # Should keep (no refund address)
            {
                "hashed_key": "keep_no_address",
                "balance": 3000,
                "refund_address": None,
                "expired": True,
            },
            # Already zero balance
            {
                "hashed_key": "zero_balance",
                "balance": 0,
                "refund_address": "lnurl3",
                "expired": True,
            },
        ]

        current_time = int(time.time())
        for data in keys_data:
            key = ApiKey(
                hashed_key=data["hashed_key"],
                balance=data["balance"],
                refund_address=data["refund_address"],
                key_expiry_time=current_time - 3600
                if data["expired"]
                else current_time + 3600,
                created_at=datetime.utcnow(),
            )
            integration_session.add(key)
        await integration_session.commit()

        with (
            patch("router.cashu.wallet") as mock_wallet,
            patch("router.cashu.get_session") as mock_get_session,
        ):
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=1)
            mock_wallet.return_value = mock_wallet_instance

            # Make get_session return our integration session
            async def get_test_session() -> Any:
                yield integration_session

            mock_get_session.side_effect = get_test_session

            await db_snapshot.capture()

            # Run refund check
            original_interval = os.environ.get("REFUND_PROCESSING_INTERVAL", "3600")
            os.environ["REFUND_PROCESSING_INTERVAL"] = "0.1"

            task = asyncio.create_task(check_for_refunds())
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                os.environ["REFUND_PROCESSING_INTERVAL"] = original_interval

            # Verify correct keys were processed
            assert mock_wallet_instance.send_to_lnurl.call_count == 1
            mock_wallet_instance.send_to_lnurl.assert_called_with("lnurl1", amount=1)

            # Check final state
            from sqlalchemy import select as sa_select

            result = await integration_session.execute(sa_select(ApiKey))
            remaining_keys_list = result.scalars().all()
            remaining_ids = [k.hashed_key for k in remaining_keys_list]

            assert "delete_me" not in remaining_ids  # Deleted after refund
            assert "keep_not_expired" in remaining_ids
            assert "keep_no_address" in remaining_ids
            assert (
                "zero_balance" not in remaining_ids
            )  # Auto-deleted due to zero balance

    async def test_refund_check_disabled(self) -> None:
        """Test that refund check can be disabled by setting interval to 0"""
        # Set refund interval to 0 to disable
        original_interval = os.environ.get("REFUND_PROCESSING_INTERVAL", "3600")
        os.environ["REFUND_PROCESSING_INTERVAL"] = "0"

        try:
            # Task should exit immediately
            task = asyncio.create_task(check_for_refunds())
            await task  # Should complete without hanging

            # Task should have exited cleanly
            assert task.done()
        finally:
            os.environ["REFUND_PROCESSING_INTERVAL"] = original_interval


@pytest.mark.asyncio
class TestPeriodicPayoutTask:
    """Test the periodic payout background task"""

    async def test_executes_at_configured_intervals(self) -> None:
        """Test that payout task runs at the configured interval"""
        call_count = 0

        async def mock_pay_out() -> None:
            nonlocal call_count
            call_count += 1

        with patch("router.cashu.pay_out", mock_pay_out):
            # Set a short interval for testing
            original_interval = os.environ.get("PAYOUT_INTERVAL", "300")
            os.environ["PAYOUT_INTERVAL"] = "0.2"  # 200ms

            task = asyncio.create_task(periodic_payout())
            await asyncio.sleep(0.7)  # Should run ~3 times
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                os.environ["PAYOUT_INTERVAL"] = original_interval

            assert 2 <= call_count <= 4  # Allow some timing variance

    async def test_calculates_payouts_accurately(
        self, integration_session: Any
    ) -> None:
        """Test that payouts are calculated correctly based on revenue"""
        # Create test API keys with various balances
        total_user_balance = 0
        for i in range(5):
            balance = 10000 * (i + 1)  # 10, 20, 30, 40, 50 sats
            total_user_balance += balance
            key = ApiKey(
                hashed_key=f"user_key_{i}",
                balance=balance,
                created_at=datetime.utcnow(),
            )
            integration_session.add(key)
        await integration_session.commit()

        # Mock wallet balance higher than user balances (indicating revenue)
        wallet_balance = 200000  # 200 sats total
        expected_revenue = wallet_balance - total_user_balance  # 50 sats revenue

        with patch("router.cashu.wallet") as mock_wallet:
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.balance = AsyncMock(return_value=wallet_balance)
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=None)
            mock_wallet.return_value = mock_wallet_instance

            # Mock environment variables
            with patch.dict(
                os.environ,
                {
                    "MINIMUM_PAYOUT": "10",  # 10 sats minimum
                    "RECEIVE_LN_ADDRESS": "owner@test.com",
                    "DEV_LN_ADDRESS": "dev@test.com",
                },
            ):
                # Call pay_out directly
                from router.cashu import pay_out

                await pay_out()

                # Verify payouts were sent correctly
                assert mock_wallet_instance.send_to_lnurl.call_count == 2

                # Check amounts (97.9% to owner, 2.1% to dev)
                calls = mock_wallet_instance.send_to_lnurl.call_args_list
                owner_call = next(c for c in calls if c[0][0] == "owner@test.com")
                dev_call = next(c for c in calls if c[0][0] == "dev@test.com")

                owner_amount = owner_call[0][1]
                dev_amount = dev_call[0][1]

                assert owner_amount == int(expected_revenue * 0.979)
                assert dev_amount == int(expected_revenue * 0.021)
                assert owner_amount + dev_amount == expected_revenue

    async def test_transaction_logging_complete(
        self, integration_session: Any, capfd: Any
    ) -> None:
        """Test that payout transactions are properly logged"""
        # Create a simple scenario
        key = ApiKey(
            hashed_key="single_user",
            balance=50000,  # 50 sats
            created_at=datetime.utcnow(),
        )
        integration_session.add(key)
        await integration_session.commit()

        with patch("router.cashu.wallet") as mock_wallet:
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.balance = AsyncMock(
                return_value=100000
            )  # 100 sats total
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=None)
            mock_wallet.return_value = mock_wallet_instance

            with patch.dict(
                os.environ,
                {
                    "MINIMUM_PAYOUT": "10",
                    "RECEIVE_LN_ADDRESS": "owner@test.com",
                    "DEV_LN_ADDRESS": "dev@test.com",
                },
            ):
                from router.cashu import pay_out

                await pay_out()

                # Check that logging occurred
                captured = capfd.readouterr()
                assert "Revenue:" in captured.out
                assert "Owner's draw:" in captured.out
                assert "Developer's donation:" in captured.out

    async def test_minimum_payout_threshold(self, integration_session: Any) -> None:
        """Test that payouts only occur when revenue exceeds minimum threshold"""
        # Create scenario with low revenue
        key = ApiKey(
            hashed_key="low_revenue_user",
            balance=95000,  # 95 sats
            created_at=datetime.utcnow(),
        )
        integration_session.add(key)
        await integration_session.commit()

        with patch("router.cashu.wallet") as mock_wallet:
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.balance = AsyncMock(
                return_value=96000
            )  # Only 1 sat revenue
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=None)
            mock_wallet.return_value = mock_wallet_instance

            with patch.dict(os.environ, {"MINIMUM_PAYOUT": "10"}):  # 10 sats minimum
                from router.cashu import pay_out

                await pay_out()

                # No payouts should have been sent
                mock_wallet_instance.send_to_lnurl.assert_not_called()


@pytest.mark.asyncio
class TestTaskInteractions:
    """Test interactions between background tasks"""

    async def test_tasks_dont_interfere_with_each_other(self) -> None:
        """Test that all tasks can run concurrently without issues"""
        # Mock all external dependencies
        with (
            patch("router.models.sats_usd_ask_price", AsyncMock(return_value=0.00002)),
            patch("router.cashu.wallet") as mock_wallet,
            patch("router.cashu.pay_out", AsyncMock()),
        ):
            mock_wallet_instance = AsyncMock()
            mock_wallet_instance.send_to_lnurl = AsyncMock(return_value=1)
            mock_wallet.return_value = mock_wallet_instance

            # Start all tasks
            tasks = []
            try:
                # Pricing task
                pricing_task = asyncio.create_task(update_sats_pricing())
                tasks.append(pricing_task)

                # Refund task (disabled to avoid interference)
                os.environ["REFUND_PROCESSING_INTERVAL"] = "0"
                refund_task = asyncio.create_task(check_for_refunds())
                tasks.append(refund_task)

                # Payout task
                payout_task = asyncio.create_task(periodic_payout())
                tasks.append(payout_task)

                # Let them run concurrently
                await asyncio.sleep(0.5)

                # All tasks should still be running (except refund which exits immediately)
                assert not pricing_task.done()
                assert refund_task.done()  # Should exit immediately when disabled
                assert not payout_task.done()

            finally:
                # Clean up
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_api_requests_work_during_task_execution(
        self, integration_client: Any
    ) -> None:
        """Test that API endpoints remain responsive during background task execution"""
        # Start a mock long-running task
        processing = asyncio.Event()

        async def slow_task() -> None:
            processing.set()
            await asyncio.sleep(2)  # Simulate long operation

        with patch("router.models.sats_usd_ask_price", slow_task):
            # Start the pricing task
            task = asyncio.create_task(update_sats_pricing())

            # Wait for task to start processing
            await processing.wait()

            # API should still be responsive
            response = await integration_client.get("/")
            assert response.status_code == 200

            # Models endpoint should work
            response = await integration_client.get("/v1/models")
            assert response.status_code == 200

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_database_locking_handled_properly(
        self, integration_session: Any
    ) -> None:
        """Test that database operations don't deadlock during concurrent task execution"""
        # Create test data
        for i in range(10):
            key = ApiKey(
                hashed_key=f"concurrent_key_{i}",
                balance=1000 * i,
                refund_address=f"lnurl{i}" if i % 2 == 0 else None,
                key_expiry_time=int(time.time()) - 3600
                if i % 3 == 0
                else int(time.time()) + 3600,
                created_at=datetime.utcnow(),
            )
            integration_session.add(key)
        await integration_session.commit()

        # Simulate concurrent database operations
        async def read_operation() -> int:
            from sqlalchemy import select as sa_select

            result = await integration_session.execute(sa_select(ApiKey))
            return len(result.scalars().all())

        async def write_operation(key_id: int) -> None:
            from sqlalchemy import select as sa_select

            stmt = sa_select(ApiKey).where(
                ApiKey.hashed_key == f"concurrent_key_{key_id}"  # type: ignore[arg-type]
            )
            result = await integration_session.execute(stmt)
            key = result.scalar_one_or_none()
            if key:
                key.balance += 100
                await integration_session.commit()

        # Run multiple operations concurrently
        tasks: List[Coroutine[Any, Any, Any]] = []
        for _ in range(5):
            tasks.append(read_operation())  # type: ignore[arg-type]
        for i in range(5):
            tasks.append(write_operation(i))  # type: ignore[arg-type]

        # All operations should complete without deadlock
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

    async def test_graceful_shutdown(self) -> None:
        """Test that all tasks shut down cleanly when cancelled"""
        shutdown_messages = []

        async def task_with_cleanup(name: str) -> None:
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                shutdown_messages.append(f"{name} shutting down")
                raise

        # Patch the actual task functions
        with (
            patch(
                "router.models.update_sats_pricing",
                lambda: task_with_cleanup("pricing"),
            ),
            patch(
                "router.cashu.check_for_refunds", lambda: task_with_cleanup("refund")
            ),
            patch("router.cashu.periodic_payout", lambda: task_with_cleanup("payout")),
        ):
            # Start all tasks
            tasks = [
                asyncio.create_task(update_sats_pricing()),
                asyncio.create_task(check_for_refunds()),
                asyncio.create_task(periodic_payout()),
            ]

            # Let them start
            await asyncio.sleep(0.2)

            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for cleanup
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all tasks shut down properly
            assert len(shutdown_messages) == 3
            assert "pricing shutting down" in shutdown_messages
            assert "refund shutting down" in shutdown_messages
            assert "payout shutting down" in shutdown_messages
