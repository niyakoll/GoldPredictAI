import { StyleSheet, Text, View } from 'react-native';

export default function CommunityScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Community</Text>
      <Text style={styles.message}>Top traders & signals coming soon...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1e', justifyContent: 'center', alignItems: 'center' },
  title: { fontSize: 32, fontWeight: '800', color: '#fff' },
  message: { fontSize: 18, color: '#888', marginTop: 20 },
});