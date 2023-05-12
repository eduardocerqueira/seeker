#date: 2023-05-12T16:41:13Z
#url: https://api.github.com/gists/2941a3817864ddadcc6e1098a3475b3f
#owner: https://api.github.com/users/zed-wong

npx hardhat verify --network mvm "0xe06aD15c6441850313D30692B4177CabbeE435F5"
npx hardhat verify --network mvm "0x2aaf3c1019FcaC45CBf00171c80aA89c88c243Db"
npx hardhat verify --network mvm "0xB269b926d06186dA332DED7d9229becfdbDA6b72"
npx hardhat verify --network mvm "0xA71697E970c5AC213D69853fc12EABcdD7B7231f" "0xB269b926d06186dA332DED7d9229becfdbDA6b72" "0x0000000000000000000000000000000000000000"
# PublicResolver: ENSRegistry NULL_ADDRESS
npx hardhat verify --network mvm "0x0E0317C7212dD4823A9CCE526A6c52F038fEAF80"
npx hardhat verify --network mvm "0xeBF8738a3De1b3Dd846613bfBFC030E921CbDA03"
npx hardhat verify --network mvm "0xc07A25Cbd7b778352b32Fc9B066Cb95616C8D5df"
npx hardhat verify --network mvm "0x1BF09A650d53324b3617dE71Aa455E2b92399EfC"
npx hardhat verify --network mvm "0x94A72fbF4527c8942563434f79B34A3e95140C9B"
npx hardhat verify --network mvm "0x67B3d068a0012482022AeEc90876594A2cCd83e9"
npx hardhat verify --network mvm "0x36EC44c369e404e6F79fC5214568153Aca8F48b5"
npx hardhat verify --network mvm "0x5B23CF4aCEaA5009DaAed8280bDfEad78F920E84" "0x2aaf3c1019FcaC45CBf00171c80aA89c88c243Db" "0xeBF8738a3De1b3Dd846613bfBFC030E921CbDA03"
# PluginRepoRegistry: managingDAO DAO_ENSSubdomainRegistrar
npx hardhat verify --network mvm "0x095BA25756fDb75Dd9Db60ef30D7c2ae39C3C357" "0x5B23CF4aCEaA5009DaAed8280bDfEad78F920E84"
# PluginRepoFactory: PluginRepoRegistry
npx hardhat verify --network mvm "0x1CE37280E8B7e4007FAF0394b6a5f20009FF3e25" "0x5B23CF4aCEaA5009DaAed8280bDfEad78F920E84"
# PluginSetupProcessor: PluginRepoRegistry
npx hardhat verify --network mvm "0x3C1D22EeD439ACA19b6844a37ca20EBf625f1511" "0x67B3d068a0012482022AeEc90876594A2cCd83e9" "0x1CE37280E8B7e4007FAF0394b6a5f20009FF3e25"
# DAOFactory: DAORegistry PluginSetupProcessor
npx hardhat verify --network mvm "0xEE5E104F0ea1Ab72b48455c7063c7Fab505348fe"
npx hardhat verify --network mvm "0x6eb0babb94CB5CDDdFdC8eD6d7bDB9De3349F2b2"
npx hardhat verify --network mvm "0x0eBaFA27d7017C938802313eEf19BD003EABe057"
npx hardhat verify --network mvm "0xDB95A078bEAF2cc00d74BE8aC55eAcb7A252a2F5"