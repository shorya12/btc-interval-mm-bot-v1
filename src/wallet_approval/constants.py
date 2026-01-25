"""Contract addresses and constants for Polygon/Polymarket."""

# Chain IDs
POLYGON_CHAIN_ID = 137
POLYGON_MUMBAI_CHAIN_ID = 80001

# Polymarket Contract Addresses (Polygon Mainnet)
# USDC.e (bridged USDC on Polygon) - used by Polymarket
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Conditional Tokens Framework (CTF)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Polymarket Exchange (main)
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Negative Risk CTF Exchange
NEG_RISK_EXCHANGE_ADDRESS = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# Negative Risk Adapter (for neg risk markets)
NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# All spender addresses that need USDC and CTF approvals
SPENDER_ADDRESSES = [
    EXCHANGE_ADDRESS,           # Main exchange
    NEG_RISK_EXCHANGE_ADDRESS,  # Neg risk exchange
    NEG_RISK_ADAPTER_ADDRESS,   # Neg risk adapter
]

# CLOB API Endpoints
CLOB_HOST_MAINNET = "https://clob.polymarket.com"
CLOB_HOST_TESTNET = "https://clob-staging.polymarket.com"

# ERC20 ABI (minimal for approvals)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]

# CTF ABI (minimal for approvals)
CTF_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_operator", "type": "address"},
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_operator", "type": "address"},
            {"name": "_approved", "type": "bool"},
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]

# Maximum uint256 for unlimited approval
MAX_UINT256 = 2**256 - 1

# USDC has 6 decimals
USDC_DECIMALS = 6
