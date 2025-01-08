#date: 2025-01-08T16:51:04Z
#url: https://api.github.com/gists/01546bfd5fc5f49ee483fef19fc87b57
#owner: https://api.github.com/users/Jorge-Lopes

$ /Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/.bin/agoric run packages/builders/scripts/vats/terminate-kread-instance.js
agoric: run: running /Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/builders/scripts/vats/terminate-kread-instance.js
agoric: run: Deploy script will run with Node.js ESM
agoric: (TypeError#1)
TypeError#1: Failed to load module "./scripts/vats/terminate-kread-instance.js" in package "file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/builders/" (1 underlying failures: Cannot find external module "@agoric/deploy-script-support" in package file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/builders/
  at throwAggregateError (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/ses/src/module-load.js:557:11)
  at load (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/ses/src/module-load.js:605:3)
  at async digestFromMap (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/@endo/compartment-mapper/src/archive-lite.js:366:3)
  at async makeAndHashArchiveFromMap (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/@endo/compartment-mapper/src/archive-lite.js:417:52)
  at async bundleZipBase64 (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/@endo/bundle-source/src/zip-base64.js:92:29)
  at async file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/deploy-script-support/src/cachedBundleSpec.js:13:20
  at async file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/deploy-script-support/src/writeCoreEvalParts.js:137:22
  at async publishRef (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/deploy-script-support/src/writeCoreEvalParts.js:152:36)
  at async writeCoreEval (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/deploy-script-support/src/writeCoreEvalParts.js:172:31)
  at async default (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/builders/scripts/vats/terminate-kread-instance.js:64:3)
  at async file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/agoric-cli/src/scripts.js:197:7
  at async Command.<anonymous> (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/agoric-cli/src/main.js:294:7)
  at async Command.parseAsync (/Users/jorgelopes/Projects/Agoric/agoric-sdk/node_modules/commander/lib/command.js:1092:5)
  at async main (file:///Users/jorgelopes/Projects/Agoric/agoric-sdk/packages/agoric-cli/src/main.js:402:5)

error Command failed with exit code 2.