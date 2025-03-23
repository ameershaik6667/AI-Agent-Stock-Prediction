import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from UI.NARmain import get_current_price, main


class TestTradingApp(unittest.TestCase):

    @patch("main.yf.Ticker")
    def test_get_current_price(self, mock_ticker):
        """Test if current stock price is fetched correctly"""
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = pd.DataFrame({"Close": [150.25]})

        price = get_current_price("AAPL")
        self.assertEqual(price, 150.25)

    @patch("builtins.input", side_effect=["test_user", "AAPL", "10", "145.50", "DONE", "1", "US", "no"])
    @patch("main.get_current_price", return_value=150.00)
    @patch("main.Crew.kickoff", return_value="Mock Response")
    def test_main_portfolio_input(self, mock_crew, mock_price, mock_input):
        """Test if user can enter portfolio details correctly"""
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_any_call("Current market price for AAPL: $150.0")
            mock_print.assert_any_call("\nResponse:")
            mock_print.assert_any_call("Mock Response")

    @patch("builtins.input", side_effect=["test_user", "DONE", "2", "no"])
    @patch("main.Crew.kickoff", return_value="Mock Signal Response")
    def test_main_signal_generation(self, mock_crew, mock_input):
        """Test if signal generation runs correctly"""
        with patch("builtins.print") as mock_print:
            main()
            mock_print.assert_any_call("\nResponse:")
            mock_print.assert_any_call("Mock Signal Response")

    @patch("builtins.input", side_effect=["test_user", "AAPL", "10", "145.50", "DONE", "2", "no"])
    @patch("main.get_current_price", return_value=150.00)
    @patch("main.Crew.kickoff", return_value="Mock Response")  # Ensure kickoff() is called
    def test_main_portfolio_input(self, mock_crew, mock_price, mock_input):
        """Test if user can enter portfolio details correctly"""
        with patch("builtins.print") as mock_print:
            main()
            
            # Verify Crew.kickoff() was actually called
            mock_crew.assert_called_once()
            
            # Ensure response is printed
            mock_print.assert_any_call("\nResponse:")
            mock_print.assert_any_call("Mock Response")


if __name__ == "__main__":
    unittest.main()
