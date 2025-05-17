import streamlit as st
import pandas as pd
from test import trade
import plotly.graph_objects as go

def create_streamlit_app():
    st.title("StockSentinel")

    file_dict = {"DJIA (USA)": "C:/Users/RUTHVIK REDDY/StockSentinel/data/^DJI_test_data.csv",
                 "NSEI (Indian)": "C:/Users/RUTHVIK REDDY/StockSentinel/data/^NSEI_test_data.csv",
                 "Apple Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/AAPL_data.csv",
                 "American Express Co": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/AXP_data.csv",
                 "Caterpillar Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/CAT_data.csv",
                 "Salesforce Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/CRM_data.csv",
                 "Goldman Sachs Group Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/GS_data.csv",
                 "Home Depot Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/HD_data.csv",
                 "Honeywell International Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/HON_data.csv",
                 "JPMorgan Chase & Co": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/JPM_data.csv",
                 "Coca-Cola Co": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/KO_data.csv",
                 "McDonald's Corp": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/MCD_data.csv",
                 "Microsoft Corp": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/MSFT_data.csv",
                 "Travelers Companies Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/TRV_data.csv",
                 "Walmart Inc": "C:/Users/RUTHVIK REDDY/StockSentinel/data/stocks_data/WMT_data.csv",
                 "NSEI (Long Term)": "C:/Users/RUTHVIK REDDY/StockSentinel/data/^NSEI_data.csv",
                 "DJIA (Long term)": "C:/Users/RUTHVIK REDDY/StockSentinel/data/DJI_data.csv"
                 }

    # model_dict = {"14M": "C:/Users/RUTHVIK REDDY/StockSentinel/models/PPO/14M_01.zip"}

    selected_key = st.selectbox("Choose test data: ", options=list(file_dict.keys()))
    test_data = file_dict[selected_key]
    # selected_model = st.selectbox("Model: ", options=list(model_dict.keys()))
    # model = model_dict["14M"]
    # episodes = st.slider("Episodes", 1, 20, 1)
    episodes = 1

    df = pd.read_csv(test_data)

    submit_button = st.button("Run")

    if submit_button:
        with st.spinner("Trading... Training... "):
            cumulative_return, max_earning_rate, max_pullback, avg_profitability_per_trade, sharpe_ratio, sortino_ratio, max_drawdown, average_profitability, net_worth_logs, rewards_list = trade(df, episodes)
            st.success("Done!")

        ## Plot Net Worth Logs
        fig_net_worth = go.Figure()

        for i, episode_log in enumerate(net_worth_logs):
            fig_net_worth.add_trace(go.Scatter(
                x=list(range(len(episode_log))),
                y=episode_log,
                mode='lines',
                name=f'Episode {i+1} Net Worth'
            ))

        fig_net_worth.update_layout(
            title=f'{selected_key} Net Worth Over Episodes',
            xaxis_title='Steps',
            yaxis_title='Net Worth',
            legend_title='Episode'
        )

        st.plotly_chart(fig_net_worth, use_container_width=True)

        ## close price
        fig_close = go.Figure()

        fig_close.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='red')
        ))

        fig_close.update_layout(
            title='Close Price Over Time',
            xaxis_title='Steps',
            yaxis_title='Close Price',
            legend_title='Price'
        )

        st.plotly_chart(fig_close, use_container_width=True)


        ## rewards
        fig = go.Figure()

        for i, episode_rewards in enumerate(rewards_list):
            fig.add_trace(go.Scatter(
                x=list(range(len(episode_rewards))),
                y=episode_rewards,
                mode='lines',
                name=f'Episode {i+1}'
            ))

        fig.update_layout(
            title=f'Rewards Over Episodes',
            xaxis_title='Steps',
            yaxis_title='Reward'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            pd.DataFrame(
                {
                    "Metrics": [
                        "Cumulative Returns",
                        "Max Earning Rate",
                        "Avg Profitability Per Trade",
                        "Sharpe Ratio",
                        "Sortino Ratio",
                        "Max Drawdown",
                        "Average Profitability"
                    ],
                    "Values": [
                        cumulative_return,
                        max_earning_rate,
                        avg_profitability_per_trade,
                        sharpe_ratio,
                        sortino_ratio,
                        max_drawdown,
                        average_profitability
                    ]
                }
            )
        )

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Model Test")
    create_streamlit_app()
