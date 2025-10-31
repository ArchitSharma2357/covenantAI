// craco.config.js
const ReactRefreshWebpackPlugin = require('@pmmmwh/react-refresh-webpack-plugin');

module.exports = {
  webpack: {
    configure: (webpackConfig, { env, paths }) => {
      if (env === 'development') {
        // Add React Refresh plugin in development
        webpackConfig.plugins.push(
          new ReactRefreshWebpackPlugin({
            overlay: false,
          })
        );
      }
      return {
        ...webpackConfig,
        module: {
          ...webpackConfig.module,
          rules: [
            ...webpackConfig.module.rules,
            {
              test: /\.[jt]sx?$/,
              exclude: /node_modules/,
              use: [
                {
                  loader: require.resolve('babel-loader'),
                  options: {
                    plugins: [
                      env === 'development' && require.resolve('react-refresh/babel')
                    ].filter(Boolean),
                  },
                },
              ],
            },
          ],
        },
      };
    },
  },
};
