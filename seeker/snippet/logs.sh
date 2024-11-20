#date: 2024-11-20T16:56:43Z
#url: https://api.github.com/gists/4a7e0e952ebd9884b422eb0843b130bd
#owner: https://api.github.com/users/Jorge-Lopes

2024-11-20T16:44:29.299Z SwingSet: vat: v1: evaluateBundleCap { manifestBundleRef: { bundleID: 'b1-df5452180466cebad460e201b1e5c360f896e30f5e758da1240436d2ed47ec1f12e525f728c9cbea2d814e8f998333292b5bb49f1b52adaa2b5adab95916158b' }, manifestGetterName: 'getManifestForUpgradeVaultFactory', vatAdminSvc: Promise [Promise] {} }
2024-11-20T16:44:29.362Z SwingSet: vat: v1: execute { manifestGetterName: 'getManifestForUpgradeVaultFactory', bundleExports: [ 'getManifestForUpgradeVaultFactory', 'upgradeVaultFactory' ] }
2024-11-20T16:44:29.377Z SwingSet: vat: v1: installation VaultFactory: new Promise
2024-11-20T16:44:29.379Z SwingSet: vat: v1: installation VaultFactory settled; remaining: []
2024-11-20T16:44:29.483Z SwingSet: vat: v1: coreProposal: upgradeVaultFactory
2024-11-20T16:44:29.483Z SwingSet: vat: v1: LOG: upgrading VaultFactory
2024-11-20T16:44:30.022Z SwingSet: vat: v1: LOG: initialPoserInvitation  Object [Alleged: Zoe Invitation payment] {}
2024-11-20T16:44:30.022Z SwingSet: vat: v1: LOG: adminFacet  Object [Alleged: adminFacet] {}
2024-11-20T16:44:30.023Z SwingSet: vat: v1: LOG: privateArgs  { feeMintAccess: Object [Alleged: FeeMint feeMintAccess] {}, initialPoserInvitation: Object [Alleged: Zoe Invitation payment] {}, initialShortfallInvitation: Object [Alleged: Zoe Invitation payment] {}, marshaller: Object [Alleged: Board readonlyMarshaller] {}, storageNode: Object [Alleged: ChainStorageNode] {} }
2024-11-20T16:44:30.023Z SwingSet: vat: v1: LOG: newPrivateArgs  { feeMintAccess: Object [Alleged: FeeMint feeMintAccess] {}, initialPoserInvitation: Object [Alleged: Zoe Invitation payment] {}, initialShortfallInvitation: Object [Alleged: Zoe Invitation payment] {}, marshaller: Object [Alleged: Board readonlyMarshaller] {}, storageNode: Object [Alleged: ChainStorageNode] {} }
2024-11-20T16:44:30.023Z SwingSet: vat: v1: LOG: vaultBundleRef  { bundleID: 'b1-90ce0f13d4251f4a2944d2653f684136130852e98f7b85e8a8bdaaa12555b82d7a60243cc7536ba3f328cc49216a7baeaae5e0da88c243f60437dc9b91ee31ec' }
2024-11-20T16:44:30.023Z SwingSet: vat: v1: LOG: bundleID  b1-90ce0f13d4251f4a2944d2653f684136130852e98f7b85e8a8bdaaa12555b82d7a60243cc7536ba3f328cc49216a7baeaae5e0da88c243f60437dc9b91ee31ec
2024-11-20T16:44:30.035Z SwingSet: kernel: attempting to upgrade vat v48 from incarnation 2 to source b1-5ce9bb36ceb21c4af80b3c11098ac049ddfaa1898cf7a9e33d100c44f5fc68e5a14a96a5d2f9af0117929b3e5b4ab58c4a404416fb5688b03a2c0e398194e6b2
2024-11-20T16:44:31.867Z SwingSet: vat: v48: ----- VF.17  2 prepare start { feeMintAccess: Object [Alleged: FeeMint feeMintAccess] {}, initialPoserInvitation: Object [Alleged: Zoe Invitation payment] {}, initialShortfallInvitation: Object [Alleged: Zoe Invitation payment] {}, marshaller: Object [Alleged: Board readonlyMarshaller] {}, storageNode: Object [Alleged: ChainStorageNode] {} } [ 'Durable Publish Kit_kindHandle', 'Recorder_kindHandle', 'Vault Holder_kindHandle', 'VaultDirector_kindHandle', 'VaultManagerKit_kindHandle', 'Vault_kindHandle', 'collateralManagers', 'debtMint', 'director', 'rewardPoolSeat', 'shortfallInvitation', 'vault param manager parts', 'vaultManagerKits' ]
2024-11-20T16:44:31.867Z SwingSet: vat: v48: ----- VF.17  3 awaiting debtMint
2024-11-20T16:44:31.873Z SwingSet: vat: v48: ----- VF.17  4 auctioneerPublicFacet Promise [Promise] {}
2024-11-20T16:44:31.888Z SwingSet: vat: v48: ----- VF.17  5 making non-durable publishers
2024-11-20T16:44:31.905Z SwingSet: vat: v48: ----- Vault Params.5  2 reviving params, keeping Object [Alleged: ATOM brand] {} { debtLimit: { brand: Object [Alleged: IST brand] {}, value: 0n }, interestRate: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 1n } }, liquidationMargin: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 150n } }, liquidationPadding: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 25n } }, liquidationPenalty: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 1n } }, mintFee: { denominator: { brand: Object [Alleged: IST brand] {}, value: 10_000n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 50n } } }
2024-11-20T16:44:31.912Z SwingSet: vat: v48: ----- Vault Params.5  3 reviving params, keeping Object [Alleged: stATOM brand] {} { debtLimit: { brand: Object [Alleged: IST brand] {}, value: 1_000_000_000n }, interestRate: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 1n } }, liquidationMargin: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 150n } }, liquidationPadding: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 25n } }, liquidationPenalty: { denominator: { brand: Object [Alleged: IST brand] {}, value: 100n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 1n } }, mintFee: { denominator: { brand: Object [Alleged: IST brand] {}, value: 10_000n }, numerator: { brand: Object [Alleged: IST brand] {}, value: 50n } } }
2024-11-20T16:44:31.916Z SwingSet: vat: v48: ----- VK.9  2 prepareVaultKit [ 'Durable Publish Kit_kindHandle', 'Recorder_kindHandle', 'Vault Holder_kindHandle', 'VaultDirector_kindHandle', 'VaultManagerKit_kindHandle', 'Vault_kindHandle', 'collateralManagers', 'debtMint', 'director', 'rewardPoolSeat', 'shortfallInvitation', 'vault param manager parts', 'vaultManagerKits' ]
2024-11-20T16:44:32.028Z SwingSet: vat: v48: ----- VM.15  2 provideAndStartVaultManagerKits start
2024-11-20T16:44:32.030Z SwingSet: vat: v48: ----- VM.15  3 Object [Alleged: ATOM brand] {} helper.start() 11
2024-11-20T16:44:32.033Z SwingSet: vat: v48: ----- VM.15  4 helper.start() making periodNotifier
2024-11-20T16:44:32.034Z SwingSet: vat: v48: ----- VM.15  5 helper.start() starting observe periodNotifier
2024-11-20T16:44:32.034Z SwingSet: vat: v48: ----- VM.15  6 helper.start() awaiting observe storedQuotesNotifier Object [Alleged: ATOM brand] {}
2024-11-20T16:44:32.034Z SwingSet: vat: v48: ----- VM.15  7 helper.start() done
2024-11-20T16:44:32.035Z SwingSet: vat: v48: ----- VM.15  8 Object [Alleged: stATOM brand] {} helper.start() 0
2024-11-20T16:44:32.035Z SwingSet: vat: v48: ----- VM.15  9 helper.start() making periodNotifier
2024-11-20T16:44:32.036Z SwingSet: vat: v48: ----- VM.15  10 helper.start() starting observe periodNotifier
2024-11-20T16:44:32.040Z SwingSet: vat: v48: ----- VM.15  11 helper.start() awaiting observe storedQuotesNotifier Object [Alleged: stATOM brand] {}
2024-11-20T16:44:32.040Z SwingSet: vat: v48: ----- VM.15  12 helper.start() done
2024-11-20T16:44:32.040Z SwingSet: vat: v48: ----- VM.15  13 provideAndStartVaultManagerKits returning
2024-11-20T16:44:32.142Z SwingSet: vat: v48: ----- VD.16  2 Start non-durable processes
2024-11-20T16:44:32.202Z SwingSet: kernel: vat v48 upgraded from incarnation 2 to 3 with source b1-5ce9bb36ceb21c4af80b3c11098ac049ddfaa1898cf7a9e33d100c44f5fc68e5a14a96a5d2f9af0117929b3e5b4ab58c4a404416fb5688b03a2c0e398194e6b2
2024-11-20T16:44:32.964Z SwingSet: ls: v9: Logging sent error stack (Error#1)
2024-11-20T16:44:32.965Z SwingSet: ls: v9: Error#1: In "getPublicFacet" method of (ZoeService): arg 0: (an undefined) - Must be a remotable InstanceHandle, not undefined
2024-11-20T16:44:32.965Z SwingSet: ls: v9: Error: In "getPublicFacet" method of (ZoeService): arg 0: (an undefined) - Must be a remotable InstanceHandle, not undefined
 at apply ()
 at Error (/bundled-source/.../node_modules/ses/src/error/tame-error-constructor.js:60)
 at makeError (/bundled-source/.../node_modules/ses/src/error/assert.js:352)
 at throwLabeled (/bundled-source/.../node_modules/@endo/common/throw-labeled.js:26)
 at applyLabelingError (/bundled-source/.../node_modules/@endo/common/apply-labeling-error.js:43)
 at mustMatch (/bundled-source/.../node_modules/@endo/patterns/src/patterns/patternMatchers.js:591)
 at defendSyncArgs (/bundled-source/.../node_modules/@endo/exo/src/exo-tools.js:82)
 at (/bundled-source/.../node_modules/@endo/exo/src/exo-tools.js:254)
 at ()

...

2024-11-20T16:44:33.494Z SwingSet: vat: v48: 💀 vaultDirector failed to start: (RemoteError(error:liveSlots:v9#70001)#2)
2024-11-20T16:44:33.494Z SwingSet: vat: v48: RemoteError(error:liveSlots:v9#70001)#2: In "getPublicFacet" method of (ZoeService): arg 0: (an undefined) - Must be a remotable InstanceHandle, not undefined