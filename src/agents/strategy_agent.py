"""
🌙 Moon Dev's Strategy Agent
Handles all strategy-based trading decisions
"""

from src.config import *
import json
from termcolor import cprint
import anthropic
import os
import importlib
import inspect
import time
import numpy as np

# Import web state for signal logging
try:
    from src.web.state import add_signal as web_add_signal
    WEB_STATE_AVAILABLE = True
except ImportError:
    WEB_STATE_AVAILABLE = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Import exchange manager for unified trading
try:
    from src.exchange_manager import ExchangeManager
    USE_EXCHANGE_MANAGER = True
except ImportError:
    from src import nice_funcs as n
    USE_EXCHANGE_MANAGER = False

# 🎯 Strategy Evaluation Prompt - JSON format for robust parsing
STRATEGY_EVAL_PROMPT = """
You are Moon Dev's Strategy Validation Assistant 🌙

Analyze the following strategy signals and validate their recommendations:

Strategy Signals:
{strategy_signals}

Market Context:
{market_data}

Your task:
1. Evaluate each strategy signal's reasoning
2. Check if signals align with current market conditions
3. Look for confirmation/contradiction between different strategies
4. Consider risk factors

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "decisions": [
    {{"signal_index": 0, "action": "EXECUTE", "confidence": 85, "reason": "Strong momentum exhaustion with volume confirmation"}},
    {{"signal_index": 1, "action": "REJECT", "confidence": 60, "reason": "Conflicting signals, low conviction"}}
  ],
  "overall_reasoning": "Summary of analysis..."
}}

Rules:
- "action" must be exactly "EXECUTE" or "REJECT"
- "signal_index" corresponds to position in signals array (0-indexed)
- "confidence" is 0-100
- If unsure, use REJECT (better to miss a trade than risk a bad one)
- Moon Dev prioritizes risk management! 🛡️
"""

class StrategyAgent:
    def __init__(self):
        """Initialize the Strategy Agent"""
        self.enabled_strategies = []
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

        # Initialize exchange manager if available
        if USE_EXCHANGE_MANAGER:
            self.em = ExchangeManager()
            cprint(f"✅ Strategy Agent using ExchangeManager for {EXCHANGE}", "green")
        else:
            self.em = None
            cprint("✅ Strategy Agent using direct nice_funcs", "green")
        
        # Import active strategy config
        try:
            from src.config import ACTIVE_STRATEGY, PAPER_TRADING
            self.active_strategy = ACTIVE_STRATEGY
            self.paper_trading = PAPER_TRADING
        except ImportError:
            self.active_strategy = 'ramf'
            self.paper_trading = True

        if ENABLE_STRATEGIES:
            try:
                # Load strategy based on ACTIVE_STRATEGY config
                if self.active_strategy == 'ramf':
                    from src.strategies.custom.ramf_strategy import RAMFStrategy
                    self.enabled_strategies.append(RAMFStrategy())
                    cprint(f"✅ Loaded RAMF Strategy (Paper Trading: {self.paper_trading})", "green")
                elif self.active_strategy == 'multifactor':
                    from src.strategies.custom.multifactor_strategy import MultifactorStrategy
                    self.enabled_strategies.append(MultifactorStrategy())
                    cprint("✅ Loaded Multifactor Strategy", "green")
                elif self.active_strategy == 'example':
                    from src.strategies.custom.example_strategy import ExampleStrategy
                    self.enabled_strategies.append(ExampleStrategy())
                    cprint("✅ Loaded Example Strategy", "green")
                elif self.active_strategy == 'sniper':
                    from src.strategies.custom.sniper_ai_strategy import SniperAIStrategy
                    self.enabled_strategies.append(SniperAIStrategy())
                    cprint(f"✅ Loaded Sniper AI Strategy (Paper Trading: {self.paper_trading})", "green")
                else:
                    # Load all strategies if unknown
                    from src.strategies.custom.example_strategy import ExampleStrategy
                    from src.strategies.custom.multifactor_strategy import MultifactorStrategy
                    self.enabled_strategies.extend([ExampleStrategy(), MultifactorStrategy()])

                print(f"✅ Loaded {len(self.enabled_strategies)} strategies!")
                for strategy in self.enabled_strategies:
                    print(f"  • {strategy.name}")

            except Exception as e:
                print(f"⚠️ Error loading strategies: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("🤖 Strategy Agent is disabled in config.py")
        
        print(f"🤖 Moon Dev's Strategy Agent initialized with {len(self.enabled_strategies)} strategies!")

    def evaluate_signals(self, signals, market_data):
        """Have LLM evaluate strategy signals with robust JSON parsing"""
        try:
            if not signals:
                return None

            # Format signals for prompt (use NumpyEncoder for numpy types)
            signals_str = json.dumps(signals, indent=2, cls=NumpyEncoder)

            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                messages=[{
                    "role": "user",
                    "content": STRATEGY_EVAL_PROMPT.format(
                        strategy_signals=signals_str,
                        market_data=market_data if market_data else "No market data available"
                    )
                }]
            )

            response = message.content
            if isinstance(response, list):
                response = response[0].text if hasattr(response[0], 'text') else str(response[0])

            # Try to parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = response.strip()
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0].strip()

                evaluation = json.loads(json_str)

                # Convert to list of decisions for backward compatibility
                decisions = []
                for decision in evaluation.get('decisions', []):
                    action = decision.get('action', 'REJECT')
                    decisions.append(action)

                reasoning = evaluation.get('overall_reasoning', 'No reasoning provided')

                cprint("🤖 Strategy Evaluation (JSON parsed successfully):", "green")
                for i, decision in enumerate(evaluation.get('decisions', [])):
                    action = decision.get('action', 'REJECT')
                    confidence = decision.get('confidence', 0)
                    reason = decision.get('reason', 'No reason')
                    color = 'green' if action == 'EXECUTE' else 'red'
                    cprint(f"  Signal {i}: {action} (confidence: {confidence}%) - {reason}", color)

                return {
                    'decisions': decisions,
                    'reasoning': reasoning,
                    'raw_evaluation': evaluation
                }

            except json.JSONDecodeError:
                # Fallback: try old parsing method
                cprint("⚠️ JSON parsing failed, using fallback parser", "yellow")
                lines = response.split('\n')
                decisions = []

                # Look for EXECUTE or REJECT keywords in each line
                for line in lines:
                    upper_line = line.upper()
                    if 'EXECUTE' in upper_line:
                        decisions.append('EXECUTE')
                    elif 'REJECT' in upper_line:
                        decisions.append('REJECT')

                # If we found fewer decisions than signals, default to REJECT for safety
                while len(decisions) < len(signals):
                    decisions.append('REJECT')

                return {
                    'decisions': decisions[:len(signals)],
                    'reasoning': response
                }

        except Exception as e:
            cprint(f"❌ Error evaluating signals: {e}", "red")
            # Return None to trigger fallback behavior
            return None

    def get_signals(self, token):
        """Get and evaluate signals from all enabled strategies"""
        try:
            # 1. Collect signals from all strategies
            signals = []
            print(f"\n🔍 Analyzing {token} with {len(self.enabled_strategies)} strategies...")

            for strategy in self.enabled_strategies:
                # Pass symbol to strategy (RAMF and newer strategies support this)
                try:
                    signal = strategy.generate_signals(symbol=token)
                except TypeError:
                    # Fallback for strategies that don't accept symbol parameter
                    signal = strategy.generate_signals()

                if signal and signal.get('token') == token:
                    signals.append({
                        'token': signal['token'],
                        'strategy_name': strategy.name,
                        'signal': signal['signal'],
                        'direction': signal['direction'],
                        'metadata': signal.get('metadata', {})
                    })

            if not signals:
                print(f"ℹ️ No strategy signals for {token}")
                return []

            print(f"\n📊 Raw Strategy Signals for {token}:")
            for signal in signals:
                print(f"  • {signal['strategy_name']}: {signal['direction']} ({signal['signal']}) for {signal['token']}")

            # 2. Get market data for context (optional)
            market_data = None
            try:
                from src.data.ohlcv_collector import collect_token_data
                market_data = collect_token_data(token)
            except Exception as e:
                print(f"⚠️ Could not get market data: {e}")

            # 3. Try LLM evaluation (but don't fail if unavailable)
            print("\n🤖 Getting LLM evaluation of signals...")
            evaluation = self.evaluate_signals(signals, market_data)

            # 4. Filter signals based on LLM decisions OR use high-confidence signals directly
            approved_signals = []

            if evaluation:
                # LLM evaluation available - use it
                for i, signal in enumerate(signals):
                    if i < len(evaluation['decisions']):
                        decision = evaluation['decisions'][i]
                        approved = "EXECUTE" in decision.upper()
                        if approved:
                            cprint(f"✅ LLM approved {signal['strategy_name']}'s {signal['direction']} signal", "green")
                            approved_signals.append(signal)
                        else:
                            cprint(f"❌ LLM rejected {signal['strategy_name']}'s {signal['direction']} signal", "red")

                        # Log signal to web dashboard
                        if WEB_STATE_AVAILABLE:
                            try:
                                web_add_signal({
                                    'token': signal.get('token', token),
                                    'direction': signal['direction'],
                                    'confidence': round(float(signal['signal']) * 100, 1),
                                    'strategy': signal['strategy_name'],
                                    'approved': approved,
                                    'reason': decision if not approved else 'LLM approved'
                                })
                            except:
                                pass
                    else:
                        cprint(f"⚠️ No LLM decision for signal {i}, defaulting to REJECT", "yellow")
            else:
                # LLM unavailable - fallback to using high-confidence signals directly
                cprint("⚠️ LLM evaluation unavailable - using signal confidence fallback", "yellow")
                for signal in signals:
                    # Only auto-approve signals with direction != NEUTRAL and confidence >= 0.7
                    if signal['direction'] != 'NEUTRAL' and signal['signal'] >= 0.7:
                        cprint(f"✅ Auto-approved high-confidence signal: {signal['strategy_name']} {signal['direction']} ({signal['signal']})", "green")
                        approved_signals.append(signal)

                        # Log approved signal to web dashboard
                        if WEB_STATE_AVAILABLE:
                            try:
                                web_add_signal({
                                    'token': signal.get('token', token),
                                    'direction': signal['direction'],
                                    'confidence': round(float(signal['signal']) * 100, 1),
                                    'strategy': signal['strategy_name'],
                                    'approved': True,
                                    'reason': 'Auto-approved (high confidence)'
                                })
                            except:
                                pass
                    elif signal['direction'] != 'NEUTRAL':
                        cprint(f"⚠️ Signal confidence too low for auto-approval: {signal['signal']}", "yellow")

            # 5. Print final approved signals
            if approved_signals:
                cprint(f"\n🎯 Final Approved Signals for {token}:", "cyan")
                for signal in approved_signals:
                    cprint(f"  • {signal['strategy_name']}: {signal['direction']} ({signal['signal']})", "white")

                # 6. Execute approved signals
                cprint("\n💫 Executing approved strategy signals...", "cyan")
                self.execute_strategy_signals(approved_signals)
            else:
                print(f"\n⚠️ No signals approved for {token}")

            return approved_signals

        except Exception as e:
            cprint(f"❌ Error getting strategy signals: {e}", "red")
            import traceback
            traceback.print_exc()
            return []

    def combine_with_portfolio(self, signals, current_portfolio):
        """Combine strategy signals with current portfolio state"""
        try:
            final_allocations = current_portfolio.copy()
            
            for signal in signals:
                token = signal['token']
                strength = signal['signal']
                direction = signal['direction']
                
                if direction == 'BUY' and strength >= STRATEGY_MIN_CONFIDENCE:
                    print(f"🔵 Buy signal for {token} (strength: {strength})")
                    max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
                    allocation = max_position * strength
                    final_allocations[token] = allocation
                elif direction == 'SELL' and strength >= STRATEGY_MIN_CONFIDENCE:
                    print(f"🔴 Sell signal for {token} (strength: {strength})")
                    final_allocations[token] = 0
            
            return final_allocations
            
        except Exception as e:
            print(f"❌ Error combining signals: {e}")
            return None 

    def execute_strategy_signals(self, approved_signals):
        """Execute trades based on approved strategy signals"""
        try:
            if not approved_signals:
                print("⚠️ No approved signals to execute")
                return

            # Check for paper trading mode
            try:
                from src.config import PAPER_TRADING
                if PAPER_TRADING:
                    cprint("\n📝 PAPER TRADING MODE - Simulating trades...", "yellow")
                    for signal in approved_signals:
                        # Try to use strategy's paper trade method if available
                        for strategy in self.enabled_strategies:
                            if hasattr(strategy, 'execute_paper_trade'):
                                strategy.execute_paper_trade(signal)
                                break
                        else:
                            cprint(f"  [PAPER] Would execute: {signal['direction']} {signal['token']}", "yellow")
                    return
            except ImportError:
                pass

            print("\n🚀 Moon Dev executing strategy signals...")
            print(f"📝 Received {len(approved_signals)} signals to execute")
            
            for signal in approved_signals:
                try:
                    print(f"\n🔍 Processing signal: {signal}")  # Debug output
                    
                    token = signal.get('token')
                    if not token:
                        print("❌ Missing token in signal")
                        print(f"Signal data: {signal}")
                        continue
                        
                    strength = signal.get('signal', 0)
                    direction = signal.get('direction', 'NOTHING')
                    
                    # Skip USDC and other excluded tokens
                    if token in EXCLUDED_TOKENS:
                        print(f"💵 Skipping {token} (excluded token)")
                        continue
                    
                    print(f"\n🎯 Processing signal for {token}...")
                    
                    # Calculate position size based on signal strength
                    max_position = usd_size * (MAX_POSITION_PERCENTAGE / 100)
                    target_size = max_position * strength
                    
                    # Get current position value (using exchange manager if available)
                    if self.em:
                        current_position = self.em.get_token_balance_usd(token)
                    else:
                        current_position = n.get_token_balance_usd(token)

                    print(f"📊 Signal strength: {strength}")
                    print(f"🎯 Target position: ${target_size:.2f} USD")
                    print(f"📈 Current position: ${current_position:.2f} USD")

                    if direction == 'BUY':
                        if current_position < target_size:
                            print(f"✨ Executing BUY for {token}")
                            if self.em:
                                self.em.ai_entry(token, target_size)
                            else:
                                n.ai_entry(token, target_size)
                            print(f"✅ Entry complete for {token}")
                        else:
                            print(f"⏸️ Position already at or above target size")

                    elif direction == 'SELL':
                        if current_position > 0:
                            print(f"📉 Executing SELL for {token}")
                            if self.em:
                                self.em.chunk_kill(token)
                            else:
                                n.chunk_kill(token, max_usd_order_size, slippage)
                            print(f"✅ Exit complete for {token}")
                        else:
                            print(f"⏸️ No position to sell")
                    
                    time.sleep(2)  # Small delay between trades
                    
                except Exception as e:
                    print(f"❌ Error processing signal: {str(e)}")
                    print(f"Signal data: {signal}")
                    continue
                
        except Exception as e:
            print(f"❌ Error executing strategy signals: {str(e)}")
            print("🔧 Moon Dev suggests checking the logs and trying again!") 