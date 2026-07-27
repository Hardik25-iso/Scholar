import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import * as api from "../api";
import type { User } from "../api";

interface AuthState {
  user: User | null;
  loading: boolean; // true until the initial /me check resolves
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // On first load, ask the server who we are (the cookie may still be valid).
  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null)) // 401 = not logged in, expected
      .finally(() => setLoading(false));
  }, []);

  const login = async (email: string, password: string) => setUser(await api.login(email, password));
  const register = async (email: string, password: string) => setUser(await api.register(email, password));
  const logout = async () => {
    await api.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
