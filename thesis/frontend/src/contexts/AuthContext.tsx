"use client"
import { useRouter,usePathname } from 'next/navigation';
import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';

// Ensure NEXT_PUBLIC_API_URL is the base URL for the API, e.g., http://localhost:8000/api/v1
// This will remove any trailing slash from NEXT_PUBLIC_API_URL to prevent double slashes in endpoint URLs.
const rawApiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
const API_BASE_URL = rawApiBaseUrl.replace(/\/$/, '');

// Define a more specific type for avatar configuration
interface AvatarConfig {
  source: string;
  // Add other config properties if they exist
}

// Define a type for the Avatar object
interface Avatar {
  glb_url: string;
  rpm_avatar_id: string;
  config: AvatarConfig;
}

// Using the User interface defined in AvatarCreatorComponent for consistency
// If it's used more broadly, consider moving it to a dedicated types file.
interface User {
  _id: string; // Corresponds to _id in your example
  username: string;
  email: string;
  // Add other fields like avatars if returned by /users/me and needed by the app
  is_active?: boolean; // Optional as it might not always be present or needed everywhere
  is_superuser?: boolean; // Optional
  is_email_verified?: boolean; // Optional
  avatars?: Avatar[]; // Array of avatar objects
  rpm_guest_user_id?: string; // Optional
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  fetchCurrentUser: () => Promise<User|null|undefined>; // Exposed for potential manual refresh
  register: (username_param: string, email_param: string, password_param: string) => Promise<boolean>; // Returns true on success
  clearError: () => void;
  setError: (message: string | null) => void;
  requestVerificationEmail: (email: string) => Promise<{ success: boolean; message?: string }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true); // True initially to check for stored token
  const [error, setErrorState] = useState<string | null>(null); // Renamed to avoid conflict
  const router=useRouter();
  const pathname = usePathname();

  const clearError = useCallback(() => {
    setErrorState(null);
  }, []);
  useEffect(()=>{
    if(user){
      if(!user.is_email_verified){
        router.push("/verify-email-sent")
      }
    }
  },[user,pathname])
  useEffect(() => {
    // On initial load, try to set the token from localStorage if not already set (e.g. by server component)
    const storedToken = localStorage.getItem('authToken');
    if (storedToken) {
      setToken(storedToken);
    } else {
      setIsLoading(false); // No token found, finished initial loading phase.
    }
  }, []);

  const setError = (message: string | null) => {
    setErrorState(message);
  };
  
  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('authToken');
    localStorage.removeItem('authUser');
    // setErrorState(null); // Optionally clear errors on logout
    // setIsLoading(false); // isLoading should be false after logout actions are complete
  }, []);

  const fetchCurrentUser = useCallback(async () => {
    const currentToken = localStorage.getItem('authToken'); // Use token from localStorage as source of truth for this fetch
    if (!currentToken) {
      setUser(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setErrorState(null);
    try {
      const response = await fetch(`${API_BASE_URL}/users/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${currentToken}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          logout(); 
          setErrorState('Session expired. Please login again.');
        } else {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch user details.' }));
          setErrorState(errorData.detail || `Error ${response.status}: Failed to fetch user details.`);
          setUser(null); 
          localStorage.removeItem('authUser');
        }
        
      } else{
        const userData: User = await response.json();
        setUser(userData);
        localStorage.setItem('authUser', JSON.stringify(userData));
        return user;
      }
    } catch (e: any) {
      console.error("Fetch current user error:", e);
      setErrorState(e.message || 'An error occurred while fetching user data.');
      setUser(null); // Clear user data on unexpected errors
      localStorage.removeItem('authUser');
    } finally {
      setIsLoading(false);
    }
    return null; 
  }, [logout]); // Added logout to dependency array

  useEffect(() => {
    // This effect runs when 'token' state changes (e.g. after initial load from localStorage, login, or logout)
    // or when fetchCurrentUser function instance changes (it shouldn't if dependencies are stable).
    if (token) {
      fetchCurrentUser();
    } else if (!token && user) {
      // Ensure user is cleared if token becomes null (e.g. after logout)
      setUser(null);
      localStorage.removeItem('authUser');
      setIsLoading(false); // Not loading if no token
    }
  }, [token, fetchCurrentUser]); // fetchCurrentUser is useCallback wrapped

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    setErrorState(null);
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }), // Matches backend LoginRequest
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Login failed. Invalid server response.' }));
        throw new Error(errData.detail || `Login failed with status: ${response.status}`);
      }
      
      // Assuming login returns { access_token: "...", token_type: "bearer" }
      const data = await response.json(); 
      
      if (!data.access_token) {
        throw new Error("Login successful, but no access token received.");
      }

      localStorage.setItem('authToken', data.access_token);
      setToken(data.access_token); // This will trigger the useEffect to call fetchCurrentUser
      // Note: fetchCurrentUser will be called by the useEffect listening to token changes.
      // The isLoading state for fetching user data will be handled by fetchCurrentUser.
    } catch (e: any) {
      setErrorState(e.message || 'An error occurred during login.');
      // Ensure token and user are cleared if login fails
      setToken(null); 
      setUser(null);
      localStorage.removeItem('authToken');
      localStorage.removeItem('authUser');
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (username_param: string, email_param: string, password_param: string): Promise<boolean> => {
    setIsLoading(true);
    setErrorState(null);
    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: username_param, email: email_param, password: password_param }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Registration failed. Invalid server response.' }));
        setErrorState(errorData.detail || `Registration failed with status: ${response.status}. Please try again.`);
        return false;
      }
      
      // Registration successful. User is not logged in by this function.
      // Optionally, parse success message: const successData = await response.json();
      return true; // Indicate success
    } catch (err: any) {
      setErrorState(err.message || 'An unexpected error occurred during registration.');
      return false;
    }
    finally {
      setIsLoading(false);
    }
  };

  const requestVerificationEmail = async (email: string): Promise<{ success: boolean; message?: string }> => {
    // Not setting global isLoading, component can manage its own loading state for this action.
    // setErrorState(null); // Component will call clearError before this if needed.
    try {
      const response = await fetch(`${API_BASE_URL}/auth/request-verification-email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }), // Backend expects { "email": "..." }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to request new verification email.' }));
        const errorMessage = errorData.detail || `Error ${response.status}: Could not send new verification email.`;
        setErrorState(errorMessage); // Set global error state
        return { success: false, message: errorMessage };
      }
      
      const responseData = await response.json(); // Backend returns {"msg": "..."} on success
      return { success: true, message: responseData.msg || "Verification email sent successfully." };
    } catch (err: any) {
      const errorMessage = err.message || 'An unexpected error occurred while requesting a new verification email.';
      setErrorState(errorMessage); // Set global error state
      return { success: false, message: errorMessage };
    }
  };


  const isAuthenticated = !!token || !!user;

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated, isLoading, error, login, logout, fetchCurrentUser, register, clearError, setError, requestVerificationEmail }}>
      {isLoading?(
        <></>
      ):children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};