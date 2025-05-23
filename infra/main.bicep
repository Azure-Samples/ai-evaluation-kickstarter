metadata name = 'azure-ai-evaluation-kickstarter'
metadata description = 'Deploys the infrastructure for Azure AI Evaluation Kickstarter'
metadata author = 'Chris Arendt <charendt@microsoft.com>'

/* -------------------------------------------------------------------------- */
/*                                 PARAMETERS                                 */
/* -------------------------------------------------------------------------- */

@minLength(1)
@maxLength(64)
@description('Name of the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@description('Principal ID of the user running the deployment')
param azurePrincipalId string

@description('Extra tags to be applied to provisioned resources')
param extraTags object = {}

@description('Location for all resources')
@allowed(['eastus2', 'swedencentral'])
param location string = 'eastus2'

/* ------------ Optional externally provided model configuration ------------ */

// @description('Optional. Externally provided model end point')
// param plannerEndpointParam string = ''

// @description('Optional. Externally provided model deployment name')
// param plannerDeploymentNameParam string = ''

// @description('Optional. Externally provided model API version')
// param plannerApiVersionParam string = ''

// @description('Optional. Externally provided model key')
// param plannerKeyParam string = ''

/* ---------------------------- Shared Resources ---------------------------- */

@maxLength(63)
@description('Name of the log analytics workspace to deploy. If not specified, a name will be generated. The maximum length is 63 characters.')
param logAnalyticsWorkspaceName string = ''

@maxLength(255)
@description('Name of the application insights to deploy. If not specified, a name will be generated. The maximum length is 255 characters.')
param applicationInsightsName string = ''

@description('Application Insights Location')
param appInsightsLocation string = location

@description('Activate authentication if true. Defaults to false.')
param useAuthentication bool = false

@description('The auth tenant id for the frontend and backend app (leave blank in AZD to use your current tenant)')
param authTenantId string = '' // Make sure authTenantId is set if not using AZD

@description('Name of the authentication client secret in the key vault')
param authClientSecretName string = 'AZURE-AUTH-CLIENT-SECRET'

@description('The auth client id for the frontend and backend app')
param authClientId string = ''

@description('Client secret of the authentication client')
@secure()
param authClientSecret string = ''

@maxLength(50)
@description('Name of the container registry to deploy. If not specified, a name will be generated. The name is global and must be unique within Azure. The maximum length is 50 characters.')
param containerRegistryName string = ''

@maxLength(60)
@description('Name of the container apps environment to deploy. If not specified, a name will be generated. The maximum length is 60 characters.')
param containerAppsEnvironmentName string = ''

/* -------------------------------- Frontend -------------------------------- */

@maxLength(32)
@description('Name of the frontend container app to deploy. If not specified, a name will be generated. The maximum length is 32 characters.')
param frontendContainerAppName string = ''

@description('Set if the frontend container app already exists.')
param frontendExists bool = false

/* -------------------------------------------------------------------------- */
/*                                  VARIABLES                                 */
/* -------------------------------------------------------------------------- */

// Load abbreviations from JSON file
var abbreviations = loadJsonContent('./abbreviations.json')

@description('Generate a unique token to make global resource names unique')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))

@description('Name of the environment with only alphanumeric characters. Used for resource names that require alphanumeric characters only')
var alphaNumericEnvironmentName = replace(replace(environmentName, '-', ''), ' ', '')

@description('Tags to be applied to all provisioned resources')
var tags = union(
  {
    'azd-env-name': environmentName
    solution: 'azure-ai-evaluation-kickstarter'
  },
  extraTags
)

@description('Azure OpenAI API Version')
var azureOpenAiApiVersion = '2024-12-01-preview'


/* -------------------------------------------------------------------------- */
/*                                  AI MODELS                                 */
/* -------------------------------------------------------------------------- */
@description('Model deployment configurations')
var deployments = [
  {
    name: 'gpt-4.1'
    sku: {
      name: 'GlobalStandard'
      capacity: 50
    }
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1'
      version: '2025-04-14'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
  {
    name: 'gpt-4.1-mini'
    sku: {
      name: 'GlobalStandard'
      capacity: 50
    }
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'
      version: '2025-04-14'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
  {
    name: 'gpt-4.1-nano'
    sku: {
      name: 'GlobalStandard'
      capacity: 50
    }
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-nano'
      version: '2025-04-14'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
  {
    name: 'gpt-4o'
    sku: {
      name: 'GlobalStandard'
      capacity: 50
    }
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-11-20'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
  {
    name: 'gpt-4o-mini'
    sku: {
      name: 'GlobalStandard'
      capacity: 50
    }
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
]

var azureOpenAiApiEndpoint = azureOpenAi.outputs.endpoint
// var executorAzureOpenAiDeploymentName = deployments[0].name
// var utilityAzureOpenAiDeploymentName =  deployments[1].name

// var plannerAzureOpenAiApiVersion = empty(plannerApiVersionParam) ? '2024-12-01-preview' : plannerApiVersionParam
// var plannerAzureOpenAiApiEndpoint = empty(plannerEndpointParam) ? azureOpenAi.outputs.endpoint : plannerEndpointParam
// var plannerAzureOpenAiDeploymentName = empty(plannerDeploymentNameParam) ? deployments[0].name : plannerDeploymentNameParam

/* --------------------- Globally Unique Resource Names --------------------- */

var _applicationInsightsName = !empty(applicationInsightsName)
  ? applicationInsightsName
  : take('${abbreviations.insightsComponents}${environmentName}', 255)
var _logAnalyticsWorkspaceName = !empty(logAnalyticsWorkspaceName)
  ? logAnalyticsWorkspaceName
  : take('${abbreviations.operationalInsightsWorkspaces}${environmentName}', 63)

var _storageAccountName = take(
  '${abbreviations.storageStorageAccounts}${alphaNumericEnvironmentName}${resourceToken}',
  24
)
var _azureOpenAiName = take(
  '${abbreviations.cognitiveServicesOpenAI}${alphaNumericEnvironmentName}${resourceToken}',
  63
)
var _aiHubName = take('${abbreviations.aiPortalHub}${environmentName}', 260)
var _aiProjectName = take('${abbreviations.aiPortalProject}${environmentName}', 260)

var _containerRegistryName = !empty(containerRegistryName)
  ? containerRegistryName
  : take('${abbreviations.containerRegistryRegistries}${alphaNumericEnvironmentName}${resourceToken}', 50)
var _keyVaultName = take('${abbreviations.keyVaultVaults}${alphaNumericEnvironmentName}-${resourceToken}', 24)
var _containerAppsEnvironmentName = !empty(containerAppsEnvironmentName)
  ? containerAppsEnvironmentName
  : take('${abbreviations.appManagedEnvironments}${environmentName}', 60)

/* ----------------------------- Resource Names ----------------------------- */

var _frontendIdentityName = take(
  '${abbreviations.managedIdentityUserAssignedIdentities}frontend-${environmentName}',
  32
)
var _frontendContainerAppName = !empty(frontendContainerAppName)
  ? frontendContainerAppName
  : take('${abbreviations.appContainerApps}frontend-${environmentName}', 32)

/* -------------------------------------------------------------------------- */
/*                                  RESOURCES                                 */
/* -------------------------------------------------------------------------- */

/* -------------------------------- AI Infra  ------------------------------- */

module hub 'modules/ai/hub.bicep' = {
  name: 'hub'
  params: {
    location: location
    tags: tags
    name: _aiHubName
    displayName: _aiHubName
    keyVaultId: keyVault.outputs.resourceId
    storageAccountId: storageAccount.outputs.resourceId
    containerRegistryId: containerRegistry.outputs.resourceId
    applicationInsightsId: appInsightsComponent.outputs.resourceId
    openAiName: azureOpenAi.outputs.name
    openAiConnectionName: 'aoai-connection'
    openAiContentSafetyConnectionName: 'aoai-content-safety-connection'
    authType: 'AAD'
    publicNetworkAccess: 'Enabled'
  }
}

module project 'modules/ai/project.bicep' = {
  name: 'project'
  params: {
    location: location
    tags: tags
    name: _aiProjectName
    displayName: _aiProjectName
    hubName: hub.outputs.name
  }
}

module storageAccount 'br/public:avm/res/storage/storage-account:0.15.0' = {
  name: 'storageAccount'
  scope: resourceGroup()
  params: {
    location: location
    tags: tags
    name: _storageAccountName
    kind: 'StorageV2'
    blobServices: {
      corsRules: [
        {
          allowedOrigins: [
            'https://mlworkspace.azure.ai'
            'https://ml.azure.com'
            'https://*.ml.azure.com'
            'https://ai.azure.com'
            'https://*.ai.azure.com'
            'https://mlworkspacecanary.azure.ai'
            'https://mlworkspace.azureml-test.net'
          ]
          allowedMethods: [
            'GET'
            'HEAD'
            'POST'
            'PUT'
            'DELETE'
            'OPTIONS'
            'PATCH'
          ]
          maxAgeInSeconds: 1800
          exposedHeaders: [
            '*'
          ]
          allowedHeaders: [
            '*'
          ]
        }
      ]
      containers: [
        {
          name: 'default'
          properties: {
            publicNetworkAccess: 'Enabled'
          }
          roleAssignments: [
            {
              roleDefinitionIdOrName: 'Storage Blob Data Contributor'
              principalId: frontendIdentity.outputs.principalId
              principalType: 'ServicePrincipal'
            }
          ]
        }
      ]
      deleteRetentionPolicy: {
        allowPermanentDelete: false
        enabled: false
      }
      shareDeleteRetentionPolicy: {
        enabled: true
        days: 7
      }
    }
  }
}

module azureOpenAi 'modules/ai/cognitiveservices.bicep' = {
  name: 'cognitiveServices'
  params: {
    location: location
    tags: tags
    name: _azureOpenAiName
    kind: 'AIServices'
    customSubDomainName: _azureOpenAiName
    deployments: deployments
    logAnalyticsWorkspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
    roleAssignments: [
      {
        roleDefinitionIdOrName: 'Cognitive Services OpenAI User'
        principalId: frontendIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
      }
      {
        roleDefinitionIdOrName: 'Cognitive Services OpenAI Contributor'
        principalId: azurePrincipalId
      }
    ]
  }
}

// Reference the deployed Cognitive Services account to fetch keys
resource openAiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: _azureOpenAiName
  dependsOn: [
    azureOpenAi   // ensure we don’t try to listKeys() until after the module deploys
  ]
}

// Retrieve the primary key from the deployed Azure OpenAI Cognitive Services account
var azureOpenAiPrimaryKey = openAiAccount.listKeys().key1

/* ---------------------------- Observability  ------------------------------ */

module logAnalyticsWorkspace 'br/public:avm/res/operational-insights/workspace:0.9.1' = {
  name: 'workspaceDeployment'
  params: {
    name: _logAnalyticsWorkspaceName
    location: location
    tags: tags
    dataRetention: 30
  }
}

module appInsightsComponent 'br/public:avm/res/insights/component:0.4.2' = {
  name: _applicationInsightsName
  params: {
    name: _applicationInsightsName
    location: appInsightsLocation
    workspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
  }
}

/* ------------------------ Common App Resources  -------------------------- */

module containerRegistry 'modules/app/container-registry.bicep' = {
  name: _containerRegistryName
  scope: resourceGroup()
  params: {
    location: location
    pullingIdentityNames: [
      _frontendIdentityName
    ]
    tags: tags
    name: '${abbreviations.containerRegistryRegistries}${resourceToken}'
  }
}

module containerAppsEnvironment 'br/public:avm/res/app/managed-environment:0.8.1' = {
  name: 'containerAppsEnvironment'
  params: {
    name: _containerAppsEnvironmentName
    location: location
    tags: tags
    logAnalyticsWorkspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
    daprAIConnectionString: appInsightsComponent.outputs.connectionString
    zoneRedundant: false
  }
}

module keyVault 'br/public:avm/res/key-vault/vault:0.11.0' = {
  name: 'keyVault'
  scope: resourceGroup()
  params: {
    location: location
    tags: tags
    name: _keyVaultName
    enableRbacAuthorization: true
    enablePurgeProtection: false // Set to true to if you deploy in production and want to protect against accidental deletion
    roleAssignments: [
      {
        roleDefinitionIdOrName: 'Key Vault Secrets User'
        principalId: frontendIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
      }
      {
        principalId: azurePrincipalId
        roleDefinitionIdOrName: 'Key Vault Administrator'
      }
    ]
    // Store auth client secret (if used) and Azure OpenAI key
    secrets: union(
      (useAuthentication && authClientSecret != '') ? [
        {
          name: authClientSecretName
          value: authClientSecret
        }
      ] : [],
      [
        {
          name: 'AzureOpenAIKey'
          value: azureOpenAiPrimaryKey
        }
      ]
    )
  }
}

/* ------------------------------ Frontend App ------------------------------ */

module frontendIdentity './modules/app/identity.bicep' = {
  name: 'frontendIdentity'
  scope: resourceGroup()
  params: {
    location: location
    identityName: _frontendIdentityName
  }
}

module frontendApp 'modules/app/container-apps.bicep' = {
  name: 'frontend-container-app'
  scope: resourceGroup()
  params: {
    name: _frontendContainerAppName
    tags: tags
    identityId: frontendIdentity.outputs.resourceId
    containerAppsEnvironmentName: containerAppsEnvironment.outputs.name
    containerRegistryName: containerRegistry.outputs.name
    exists: frontendExists
    serviceName: 'frontend' // Must match the service name in azure.yaml
    env: {
      // Required for the frontend app to ask for a token for the backend app
      AZURE_CLIENT_APP_ID: authClientId

      // Required for container app daprAI
      APPLICATIONINSIGHTS_CONNECTION_STRING: appInsightsComponent.outputs.connectionString

      // Required for managed identity
      AZURE_CLIENT_ID: frontendIdentity.outputs.clientId

      AZURE_SUBSCRIPTION_ID: subscription().subscriptionId
      AZURE_RESOURCE_GROUP: resourceGroup().name
      SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS: true
      SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE: true // OBS! You might want to remove this in production
      AZURE_PROJECT_NAME: _aiProjectName
      AZURE_KEY_VAULT_URL: keyVault.outputs.uri
      AZURE_OPENAI_SECRET_NAME: 'AzureOpenAIKey'
      AZURE_OPENAI_API_VERSION: azureOpenAiApiVersion
      AZURE_OPENAI_ENDPOINT: azureOpenAiApiEndpoint
      SUPPORTED_MODELS: 'gpt-4.1,gpt-4.1-mini,gpt-4.1-nano,gpt-4o,gpt-4o-mini'
    }
    keyvaultIdentities: useAuthentication
      ? {
          'microsoft-provider-authentication-secret': {
            keyVaultUrl: '${keyVault.outputs.uri}secrets/${authClientSecretName}'
            identity: frontendIdentity.outputs.resourceId
          }
        }
      : {}
  }
}

module frontendContainerAppAuth 'modules/app/container-apps-auth.bicep' = if (useAuthentication) {
  name: 'frontend-container-app-auth-module'
  params: {
    name: frontendApp.outputs.name
    clientId: authClientId
    clientSecretName: 'microsoft-provider-authentication-secret'
    openIdIssuer: '${environment().authentication.loginEndpoint}${authTenantId}/v2.0' // Works only for Microsoft Entra
    unauthenticatedClientAction: 'RedirectToLoginPage'
    allowedApplications: [
      '04b07795-8ddb-461a-bbee-02f9e1bf7b46' // AZ CLI for testing purposes
    ]
  }
}


/* -------------------------------------------------------------------------- */
/*                                   OUTPUTS                                  */
/* -------------------------------------------------------------------------- */

// Outputs are automatically saved in the local azd environment .env file.
// To see these outputs, run `azd env get-values`,  or
// `azd env get-values --output json` for json output.
// To generate your own `.env` file run `azd env get-values > .env`


/* --------------------------- Apps Deployment ----------------------------- */

@description('The endpoint of the container registry.') // necessary for azd deploy
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer

@description('Endpoint URL of the Frontend service')
output SERVICE_FRONTEND_URL string = frontendApp.outputs.URL

// @description('Endpoint URL of the Backend service')
// output SERVICE_BACKEND_URL string = backendApp.outputs.URL


/* ------------------------ Authentication & RBAC -------------------------- */

@description('Activate authentication if true')
output USE_AUTHENTICATION bool = useAuthentication

@description('ID of the tenant we are deploying to')
output AZURE_AUTH_TENANT_ID string = authTenantId

@description('Principal ID of the user running the deployment')
output AZURE_PRINCIPAL_ID string = azurePrincipalId

@description('Application registration client ID')
output AZURE_CLIENT_APP_ID string = authClientId

/* ------------------------------- Models --------------------------------- */

@description('Azure OpenAI endpoint')
output AZURE_OPENAI_ENDPOINT string = azureOpenAiApiEndpoint

@description('Azure OpenAI API Version')
output AZURE_OPENAI_API_VERSION string = azureOpenAiApiVersion

// @description('Azure OpenAI Model Deployment Name - Executor Service')
// output EXECUTOR_AZURE_OPENAI_DEPLOYMENT_NAME string = executorAzureOpenAiDeploymentName

// @description('Azure OpenAI Model Deployment Name - Utility Service')
// output UTILITY_AZURE_OPENAI_DEPLOYMENT_NAME string = utilityAzureOpenAiDeploymentName

// @description('Azure OpenAI Model Deployment Name: Planner')
// output PLANNER_AZURE_OPENAI_DEPLOYMENT_NAME string = plannerAzureOpenAiDeploymentName

// @description('Azure OpenAI endpoint: Planner')
// output PLANNER_AZURE_OPENAI_ENDPOINT string = plannerAzureOpenAiApiEndpoint

// @description('Azure OpenAI API Version: Planner')
// output PLANNER_AZURE_OPENAI_API_VERSION string = plannerAzureOpenAiApiVersion

// @description('Azure OpenAI Key: Planner')
// output plannerkeysecret string = plannerKeyParam

@description('Azure AI Project Name')
output AZURE_PROJECT_NAME string = _aiProjectName

@description('Azure Key Vault URL')
output AZURE_KEY_VAULT_URL string = keyVault.outputs.uri

@description('Azure OpenAI Secret Name in Azure Key Vault')
output AZURE_OPENAI_SECRET_NAME string = 'AzureOpenAIKey'

/* --------------------------- Observability ------------------------------ */

@description('Application Insights name')
output AZURE_APPLICATION_INSIGHTS_NAME string = appInsightsComponent.outputs.name

@description('Log Analytics Workspace name')
output AZURE_LOG_ANALYTICS_WORKSPACE_NAME string = logAnalyticsWorkspace.outputs.name

@description('Application Insights connection string')
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appInsightsComponent.outputs.connectionString

@description('Semantic Kernel Diagnostics')
output SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS bool = true

@description('Semantic Kernel Diagnostics: if set, content of the messages is traced. Set to false in production')
output SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE bool = true
