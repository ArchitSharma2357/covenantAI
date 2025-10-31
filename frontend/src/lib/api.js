const API_URL = process.env.REACT_APP_API_URL || 'https://backendcovenentai.up.railway.app';

export const getApiUrl = (path) => {
    // Remove leading slash if present
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    return `${API_URL}/${cleanPath}`;
};

export const apiConfig = {
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
};

// Add auth token to requests if available
export const addAuthHeader = (headers = {}) => {
    const token = localStorage.getItem('token');
    if (token) {
        return {
            ...headers,
            'Authorization': `Bearer ${token}`,
        };
    }
    return headers;
};
